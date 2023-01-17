/* pocl_llvm_utils.cc: various helpers for pocl LLVM API.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2020 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "config.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"
#include <unistd.h>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringMap.h>

#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Signals.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <llvm/Target/TargetMachine.h>

#include <llvm/IRReader/IRReader.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

#include <llvm/Support/raw_os_ostream.h>
#include <llvm/PassRegistry.h>

#include <llvm/IR/LegacyPassManager.h>
#define PassManager legacy::PassManager

#include <llvm/InitializePasses.h>
#include <llvm/Support/CommandLine.h>
#include <llvm-c/Core.h>

using namespace llvm;

#include <string>
#include <map>

llvm::Module *parseModuleIR(const char *path, llvm::LLVMContext *c) {
  SMDiagnostic Err;
  return parseIRFile(path, Err, *c).release();
}

void parseModuleGVarSize(cl_program program, unsigned device_i,
                         llvm::Module *ProgramBC) {

  unsigned long TotalGVarBytes = 0;
  if (!getModuleIntMetadata(*ProgramBC, PoclGVarMDName, TotalGVarBytes))
    return;

  if (TotalGVarBytes) {
    if (program->global_var_total_size[device_i])
      assert(program->global_var_total_size[device_i] == TotalGVarBytes);
    else
      program->global_var_total_size[device_i] = TotalGVarBytes;
    POCL_MSG_PRINT_LLVM("Total Global Variable Bytes: %zu\n", TotalGVarBytes);
  }
}

void writeModuleIRtoString(const llvm::Module *mod, std::string& dest) {
  llvm::raw_string_ostream sos(dest);
  WriteBitcodeToFile(*mod, sos);
  sos.str(); // flush
}

int pocl_write_module(void *module, const char *path, int dont_rewrite) {
  assert(module);
  assert(path);

  std::string binary;
  writeModuleIRtoString((const Module *)module, binary);

  return pocl_write_file(path, binary.data(), (uint64_t)binary.size(), 0,
                         dont_rewrite);
}

llvm::Module *parseModuleIRMem(const char *input_stream, size_t size,
                               llvm::LLVMContext *c) {
  StringRef input_stream_ref(input_stream, size);
  std::unique_ptr<MemoryBuffer> buffer =
      MemoryBuffer::getMemBufferCopy(input_stream_ref);

  auto parsed_module = parseBitcodeFile(buffer->getMemBufferRef(), *c);
  if (!parsed_module)
    return nullptr;
  return parsed_module.get().release();
}

static int getModuleTriple(const char *input_stream, size_t size,
                           std::string &triple) {
  StringRef input_stream_ref(input_stream, size);
  std::unique_ptr<MemoryBuffer> buffer =
      MemoryBuffer::getMemBufferCopy(input_stream_ref);
  if (!isBitcode((const unsigned char*)input_stream,
                 (const unsigned char*)input_stream+size))
    return -1;

  auto triple_e = getBitcodeTargetTriple(buffer->getMemBufferRef());
  if (!triple_e)
    return -1;
  triple = triple_e.get();
  return 0;
}

char *pocl_get_llvm_cpu_name() {
  StringRef r = llvm::sys::getHostCPUName();

  // LLVM may return an empty string -- treat as generic
  if (r.empty())
    r = "generic";

#ifndef KERNELLIB_HOST_DISTRO_VARIANTS
  if (r.str() == "generic" && strlen(OCL_KERNEL_TARGET_CPU)) {
    POCL_MSG_WARN("LLVM does not recognize your cpu, trying to use "
                   OCL_KERNEL_TARGET_CPU " for -target-cpu\n");
    r = llvm::StringRef(OCL_KERNEL_TARGET_CPU);
  }
#endif

  assert(r.size() > 0);
  char *cpu_name = (char *)malloc(r.size() + 1);
  strncpy(cpu_name, r.data(), r.size());
  cpu_name[r.size()] = 0;
  return cpu_name;
}

int bitcode_is_triple(const char *bitcode, size_t size, const char *triple) {
  std::string Triple;
  if (getModuleTriple(bitcode, size, Triple) == 0)
    return Triple.find(triple) != std::string::npos;
  else
    return 0;
}

// TODO this should be fixed to not require LLVM eventually,
// so that LLVM-less builds also report FMA correctly.
int cpu_has_fma() {
  StringMap<bool> features;
  bool res = llvm::sys::getHostCPUFeatures(features);
  return ((res && (features["fma"] || features["fma4"])) ? 1 : 0);
}

#define VECWIDTH(x)                                                            \
  std::min(std::max((lane_width / (unsigned)(sizeof(x))), 1U), 16U)

void cpu_setup_vector_widths(cl_device_id dev) {
  StringMap<bool> features;
  bool res = llvm::sys::getHostCPUFeatures(features);
  unsigned lane_width = 1;
  if (res) {
    if ((features["sse"]) || (features["neon"]))
      lane_width = 16;
    if (features["avx"])
      lane_width = 32;
    if (features["avx512f"])
      lane_width = 64;
  }

  dev->native_vector_width_char = dev->preferred_vector_width_char =
      VECWIDTH(cl_char);
  dev->native_vector_width_short = dev->preferred_vector_width_short =
      VECWIDTH(cl_short);
  dev->native_vector_width_int = dev->preferred_vector_width_int =
      VECWIDTH(cl_int);
  dev->native_vector_width_long = dev->preferred_vector_width_long =
      VECWIDTH(cl_long);
  dev->native_vector_width_float = dev->preferred_vector_width_float =
      VECWIDTH(float);
#ifdef _CL_DISABLE_DOUBLE
  dev->native_vector_width_double = dev->preferred_vector_width_double = 0;
#else
  dev->native_vector_width_double = dev->preferred_vector_width_double =
      VECWIDTH(double);
#endif

#ifdef _CL_DISABLE_HALF
  dev->native_vector_width_half = dev->preferred_vector_width_half = 0;
#else
  dev->native_vector_width_half = dev->preferred_vector_width_half =
      VECWIDTH(cl_short);
#endif
}

int pocl_llvm_remove_file_on_signal(const char *file) {
  return llvm::sys::RemoveFileOnSignal(
            StringRef(file)) ? 0 : -1;
}

/*
 * Use one global LLVMContext across all LLVM bitcodes. This is because
 * we want to cache the bitcode IR libraries and reuse them when linking
 * new kernels. The CloneModule etc. seem to assume we are linking
 * bitcodes with a same LLVMContext. Unfortunately, this requires serializing
 * all calls to the LLVM APIs with mutex.
 * Freeing/deleting the context crashes LLVM 3.2 (at program exit), as a
 * work-around, allocate this from heap.
 */

static void diagHandler(LLVMDiagnosticInfoRef DI, void *diagprinter) {
  assert(diagprinter);
  DiagnosticPrinterRawOStream *poclDiagPrinter =
      (DiagnosticPrinterRawOStream *)diagprinter;
  unwrap(DI)->print(*poclDiagPrinter);
}

std::string getDiagString(cl_context ctx) {
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;

  llvm_ctx->poclDiagStream->flush();
  std::string ret(*llvm_ctx->poclDiagString);
  llvm_ctx->poclDiagString->clear();
  return ret;
}


/* The LLVM API interface functions are not at the moment not thread safe,
 * Pocl needs to ensure only one thread is using this layer at the time.
 */
PoclCompilerMutexGuard::PoclCompilerMutexGuard(pocl_lock_t *ptr) {
  lock = ptr;
  POCL_LOCK(*lock);
}

PoclCompilerMutexGuard::~PoclCompilerMutexGuard() { POCL_UNLOCK(*lock); }

std::string currentWgMethod;

static bool LLVMInitialized = false;
static bool LLVMOptionsInitialized = false;

/* must be called with kernelCompilerLock locked */
void InitializeLLVM() {

  if (!LLVMInitialized) {

    LLVMInitialized = true;
    // We have not initialized any pass managers for any device yet.
    // Run the global LLVM pass initialization functions.
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    PassRegistry &Registry = *PassRegistry::getPassRegistry();

    initializeCore(Registry);
    initializeScalarOpts(Registry);
    initializeVectorization(Registry);
    initializeIPO(Registry);
    initializeAnalysis(Registry);
    initializeTransformUtils(Registry);
    initializeInstCombine(Registry);
    initializeInstrumentation(Registry);
    initializeTarget(Registry);
  }

  // Set the options only once. TODO: fix it so that each
  // device can reset their own options. Now one cannot compile
  // with different options to different devices at one run.

  if (!LLVMOptionsInitialized) {

    LLVMOptionsInitialized = true;

    StringMap<llvm::cl::Option *> &opts = llvm::cl::getRegisteredOptions();

    llvm::cl::Option *O = nullptr;

    currentWgMethod =
        pocl_get_string_option("POCL_WORK_GROUP_METHOD", "loopvec");

    if (currentWgMethod == "loopvec") {

      O = opts["scalarize-load-store"];
      assert(O && "could not find LLVM option 'scalarize-load-store'");
      O->addOccurrence(1, StringRef("scalarize-load-store"), StringRef("1"),
                       false);

      // LLVM inner loop vectorizer does not check whether the loop inside
      // another loop, in which case even a small trip count loops might be
      // worthwhile to vectorize.
      O = opts["vectorizer-min-trip-count"];
      assert(O && "could not find LLVM option 'vectorizer-min-trip-count'");
      O->addOccurrence(1, StringRef("vectorizer-min-trip-count"),
                       StringRef("2"), false);

      // Disable jump threading optimization with following two options from
      // duplicating blocks. Using jump threading will mess up parallel region
      // construction especially when kernel contains barriers.
      // TODO: If enabled then parallel region construction code needs
      // improvements and make sure it doesn't disallow other optimizations like
      // vectorization.
      O = opts["jump-threading-threshold"];
      assert(O && "could not find LLVM option 'jump-threading-threshold'");
      O->addOccurrence(1, StringRef("jump-threading-threshold"), StringRef("0"),
                       false);
      O = opts["jump-threading-implication-search-threshold"];
      assert(O && "could not find LLVM option "
                  "'jump-threading-implication-search-threshold'");
      O->addOccurrence(1,
                       StringRef("jump-threading-implication-search-threshold"),
                       StringRef("0"), false);

      if (pocl_get_bool_option("POCL_VECTORIZER_REMARKS", 0) == 1) {
        // Enable diagnostics from the loop vectorizer.
        O = opts["pass-remarks-missed"];
        assert(O && "could not find LLVM option 'pass-remarks-missed'");
        O->addOccurrence(1, StringRef("pass-remarks-missed"),
                         StringRef("loop-vectorize"), false);

        O = opts["pass-remarks-analysis"];
        assert(O && "could not find LLVM option 'pass-remarks-analysis'");
        O->addOccurrence(1, StringRef("pass-remarks-analysis"),
                         StringRef("loop-vectorize"), false);

        O = opts["pass-remarks"];
        assert(O && "could not find LLVM option 'pass-remarks'");
        O->addOccurrence(1, StringRef("pass-remarks"),
                         StringRef("loop-vectorize"), false);
      }
    }
    if (pocl_get_bool_option("POCL_DEBUG_LLVM_PASSES", 0) == 1) {
      O = opts["debug"];
      assert(O && "could not find LLVM option 'debug'");
      O->addOccurrence(1, StringRef("debug"), StringRef("true"), false);
    }
#if LLVM_MAJOR == 9
    O = opts["unroll-threshold"];
    assert(O && "could not find LLVM option 'unroll-threshold'");
    O->addOccurrence(1, StringRef("unroll-threshold"), StringRef("1"), false);
#endif
  }
}

/* re-initialization causes errors like this:
clang: for the   --scalarize-load-store option: may only occur zero or one
times! clang: for the   --vectorizer-min-trip-count option: may only occur zero
or one times! clang: for the   --unroll-threshold option: may only occur zero or
one times!
*/

void UnInitializeLLVM() {
  clearKernelPasses();
  clearTargetMachines();
  LLVMInitialized = false;
}

#define GLOBAL_LLVM_CONTEXT

#ifdef GLOBAL_LLVM_CONTEXT
static PoclLLVMContextData *GlobalLLVMContext = nullptr;
static unsigned GlobalLLVMContextRefcount = 0;
#endif

void pocl_llvm_create_context(cl_context ctx) {

#ifdef GLOBAL_LLVM_CONTEXT
  if (GlobalLLVMContext != nullptr) {
    ctx->llvm_context_data = GlobalLLVMContext;
    ++GlobalLLVMContextRefcount;
    return;
  }
#endif

  PoclLLVMContextData *data = new PoclLLVMContextData;
  assert(data);

  data->Context = new llvm::LLVMContext();
  assert(data->Context);
#if (CLANG_MAJOR == 15)
#ifdef LLVM_OPAQUE_POINTERS
  data->Context->setOpaquePointers(true);
#else
  data->Context->setOpaquePointers(false);
#endif
#endif
  data->number_of_IRs = 0;
  data->poclDiagString = new std::string;
  data->poclDiagStream = new llvm::raw_string_ostream(*data->poclDiagString);
  data->poclDiagPrinter =
      new DiagnosticPrinterRawOStream(*data->poclDiagStream);

  data->kernelLibraryMap = new kernelLibraryMapTy;
  assert(data->kernelLibraryMap);
  POCL_INIT_LOCK(data->Lock);

  LLVMContextSetDiagnosticHandler(wrap(data->Context),
                                  (LLVMDiagnosticHandler)diagHandler,
                                  (void *)data->poclDiagPrinter);
  assert(ctx->llvm_context_data == nullptr);
  ctx->llvm_context_data = data;
#ifdef GLOBAL_LLVM_CONTEXT
  GlobalLLVMContext = data;
  ++GlobalLLVMContextRefcount;
#endif

  POCL_MSG_PRINT_LLVM("Created context %" PRId64 " (%p)\n", ctx->id, ctx);
}

void pocl_llvm_release_context(cl_context ctx) {

  POCL_MSG_PRINT_LLVM("releasing LLVM context\n");

#ifdef GLOBAL_LLVM_CONTEXT
  --GlobalLLVMContextRefcount;
  if (GlobalLLVMContextRefcount > 0)
    return;
#endif

  PoclLLVMContextData *data = (PoclLLVMContextData *)ctx->llvm_context_data;
  assert(data);

  if (data->number_of_IRs > 0) {
    POCL_ABORT("still have references to IRs - can't release LLVM context !\n");
  }

  delete data->poclDiagPrinter;
  delete data->poclDiagStream;
  delete data->poclDiagString;

  assert(data->kernelLibraryMap);
  // void cleanKernelLibrary(cl_context ctx) {
  for (auto i = data->kernelLibraryMap->begin(),
            e = data->kernelLibraryMap->end();
       i != e; ++i) {
    delete (llvm::Module *)i->second;
  }
  data->kernelLibraryMap->clear();
  delete data->kernelLibraryMap;
  POCL_DESTROY_LOCK(data->Lock);

  delete data->Context;
  delete data;
  ctx->llvm_context_data = nullptr;
#ifdef GLOBAL_LLVM_CONTEXT
  GlobalLLVMContext = nullptr;
#endif
}

#define POCL_METADATA_ROOT "pocl_meta"

void setModuleIntMetadata(llvm::Module *mod, const char *key, unsigned long data) {

  llvm::Metadata *meta[] = {MDString::get(mod->getContext(), key),
                            llvm::ConstantAsMetadata::get(ConstantInt::get(
                                Type::getInt64Ty(mod->getContext()), data))};

  MDNode *MD = MDNode::get(mod->getContext(), meta);

  NamedMDNode *Root = mod->getOrInsertNamedMetadata(POCL_METADATA_ROOT);
  Root->addOperand(MD);
}

void setModuleStringMetadata(llvm::Module *mod, const char *key,
                             const char *data) {
  llvm::Metadata *meta[] = {MDString::get(mod->getContext(), key),
                            MDString::get(mod->getContext(), data)};

  MDNode *MD = MDNode::get(mod->getContext(), meta);

  NamedMDNode *Root = mod->getOrInsertNamedMetadata(POCL_METADATA_ROOT);
  Root->addOperand(MD);
}

void setModuleBoolMetadata(llvm::Module *mod, const char *key, bool data) {
  llvm::Metadata *meta[] = {
      MDString::get(mod->getContext(), key),
      llvm::ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt8Ty(mod->getContext()), data ? 1 : 0))};

  MDNode *MD = MDNode::get(mod->getContext(), meta);

  NamedMDNode *Root = mod->getOrInsertNamedMetadata(POCL_METADATA_ROOT);
  Root->addOperand(MD);
}

bool getModuleIntMetadata(const llvm::Module &mod, const char *key,
                          unsigned long &data) {
  NamedMDNode *Root = mod.getNamedMetadata(POCL_METADATA_ROOT);
  if (!Root)
    return false;

  bool found = false;

  for (size_t i = 0; i < Root->getNumOperands(); ++i) {
    MDNode *MD = Root->getOperand(i);

    Metadata *KeyMD = MD->getOperand(0);
    assert(KeyMD);
    MDString *Key = dyn_cast<MDString>(KeyMD);
    assert(Key);
    if (Key->getString().compare(key) != 0)
      continue;

    Metadata *ValueMD = MD->getOperand(1);
    assert(ValueMD);
    ConstantInt *CI = mdconst::extract<ConstantInt>(ValueMD);
    data = CI->getZExtValue();
    found = true;
  }
  return found;
}

bool getModuleStringMetadata(const llvm::Module &mod, const char *key,
                             std::string &data) {
  NamedMDNode *Root = mod.getNamedMetadata(POCL_METADATA_ROOT);
  if (!Root)
    return false;

  bool found = false;

  for (size_t i = 0; i < Root->getNumOperands(); ++i) {
    MDNode *MD = Root->getOperand(i);

    Metadata *KeyMD = MD->getOperand(0);
    assert(KeyMD);
    MDString *Key = dyn_cast<MDString>(KeyMD);
    assert(Key);
    if (Key->getString().compare(key) != 0)
      continue;

    Metadata *ValueMD = MD->getOperand(1);
    assert(ValueMD);
    MDString *StringValue = dyn_cast<MDString>(ValueMD);
    data = StringValue->getString().str();
    found = true;
  }
  return found;
}

bool getModuleBoolMetadata(const llvm::Module &mod, const char *key,
                           bool &data) {
  unsigned long temporary;
  bool found = getModuleIntMetadata(mod, key, temporary);
  if (found) {
    data = temporary > 0;
  }
  return found;
}
