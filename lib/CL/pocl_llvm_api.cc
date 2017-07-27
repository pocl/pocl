/* pocl_llvm_api.cc: C wrappers for calling the LLVM/Clang C++ APIs to invoke
   the different kernel compilation phases.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2017 Pekka Jääskeläinen

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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")
IGNORE_COMPILER_WARNING("-Wstrict-aliasing")

#include "config.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"

#ifndef LLVM_OLDER_THAN_4_0
#include "clang/Lex/PreprocessorOptions.h"
#endif

// For some reason including pocl.h before including CodeGenAction.h
// causes an error. Some kind of macro definition issue. To investigate.
#include "pocl.h"


#include "llvm/LinkAllPasses.h"
#include "llvm/Linker/Linker.h"
#ifdef LLVM_OLDER_THAN_3_7
#include "llvm/PassManager.h"
#include "llvm/Target/TargetLibraryInfo.h"
#else
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#define PassManager legacy::PassManager
#endif

#ifdef LLVM_OLDER_THAN_4_0
#include "llvm/Bitcode/ReaderWriter.h"
#else
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#endif

#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Linker/Linker.h"
#include "llvm/PassAnalysisSupport.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IRReader/IRReader.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/Host.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cstdio>
#include <map>

// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros, 
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm.h"
#include "pocl_runtime_config.h"
#include "install-paths.h"
#include "LLVMUtils.h"
#include "linker.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "TargetAddressSpaces.h"

using namespace clang;
using namespace llvm;

POP_COMPILER_DIAGS

/**
 * Use one global LLVMContext across all LLVM bitcodes. This is because
 * we want to cache the bitcode IR libraries and reuse them when linking
 * new kernels. The CloneModule etc. seem to assume we are linking
 * bitcodes with a same LLVMContext. Unfortunately, this requires serializing
 * all calls to the LLVM APIs with mutex.
 * Freeing/deleting the context crashes LLVM 3.2 (at program exit), as a
 * work-around, allocate this from heap.
 */
static LLVMContext *globalContext = NULL;
static bool LLVMInitialized = false;

static LLVMContext *GlobalContext() {
  if (globalContext == NULL) globalContext = new LLVMContext();
  return globalContext;
}

/* The LLVM API interface functions are not at the moment not thread safe,
   ensure only one thread is using this layer at the time with a mutex. */

static llvm::sys::Mutex kernelCompilerLock;

/* Global pocl device to be used by passes if needed */
cl_device_id currentPoclDevice = NULL;

static std::map<cl_device_id, llvm::TargetMachine *> targetMachines;
static std::map<cl_device_id, PassManager *> kernelPassesNoflatten;
static std::map<cl_device_id, PassManager *> kernelPassesFlatten;
static std::map<cl_device_id, llvm::Module *> kernelLibraryMap;
static std::string currentWgMethod;

/* must be called with kernelCompilerLock locked */
static void InitializeLLVM() {

  if (LLVMInitialized)
    return;
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
#ifdef LLVM_OLDER_THAN_3_8
  initializeIPA(Registry);
#endif
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);

// Set the options only once. TODO: fix it so that each
// device can reset their own options. Now one cannot compile
// with different options to different devices at one run.

#ifdef LLVM_OLDER_THAN_3_7
  StringMap<llvm::cl::Option *> opts;
  llvm::cl::getRegisteredOptions(opts);
#else
  StringMap<llvm::cl::Option *> &opts = llvm::cl::getRegisteredOptions();
#endif

  llvm::cl::Option *O = nullptr;

  currentWgMethod = pocl_get_string_option("POCL_WORK_GROUP_METHOD", "loopvec");

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
    O->addOccurrence(1, StringRef("vectorizer-min-trip-count"), StringRef("2"),
                     false);

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

  O = opts["unroll-threshold"];
  assert(O && "could not find LLVM option 'unroll-threshold'");
  O->addOccurrence(1, StringRef("unroll-threshold"), StringRef("1"), false);

  LLVMInitialized = true;
}

// TODO FIXME currently pocl_llvm_release() only works when
// there are zero programs with IRs, because
// programs hold references to LLVM IRs
static long numberOfIRs = 0;

void pocl_llvm_release() {

  llvm::MutexGuard lockHolder(kernelCompilerLock);

  assert(numberOfIRs >= 0);

  if (numberOfIRs > 0) {
    POCL_MSG_PRINT_LLVM("still have references to IRs - not releasing LLVM\n");
    return;
  } else {
    POCL_MSG_PRINT_LLVM("releasing LLVM\n");
  }

  for (auto i = kernelLibraryMap.begin(), e = kernelLibraryMap.end();
       i != e; ++i) {
    delete (llvm::Module *)i->second;
  }

  for (auto i = kernelPassesFlatten.begin(), e = kernelPassesFlatten.end();
       i != e; ++i) {
    PassManager *pm = (PassManager *)i->second;
    delete pm;
  }

  for (auto i = kernelPassesNoflatten.begin(),
            e = kernelPassesNoflatten.end();
       i != e; ++i) {
    PassManager *pm = (PassManager *)i->second;
    delete pm;
  }

  for (auto i = targetMachines.begin(), e = targetMachines.end(); i != e;
       ++i) {
    delete (llvm::TargetMachine *)i->second;
  }

  targetMachines.clear();
  kernelPassesFlatten.clear();
  kernelPassesNoflatten.clear();
  kernelLibraryMap.clear();
  delete globalContext;
  globalContext = nullptr;
  LLVMInitialized = false;
}

//#define DEBUG_POCL_LLVM_API

#if defined(DEBUG_POCL_LLVM_API) && defined(NDEBUG)
#undef NDEBUG
#include <cassert>
#endif


// Read input source to clang::FrontendOptions.
// The source is contained in the program->source array,
// but if debugging option is enabled in the kernel compiler
// we need to dump the file to disk first for the debugger
// to find it.
static inline int
load_source(FrontendOptions &fe,
            cl_program program)
{
  char source_file[POCL_FILENAME_LENGTH];
  POCL_RETURN_ERROR_ON(pocl_cache_write_program_source(source_file, program),
                       CL_OUT_OF_HOST_MEMORY, "Could not write program source");

  fe.Inputs.push_back
#if LLVM_OLDER_THAN_5_0
      (FrontendInputFile(source_file, clang::IK_OpenCL));
#else
      (FrontendInputFile(source_file, clang::InputKind::OpenCL));
#endif

  return 0;
}

// Unlink input sources
static inline int
unlink_source(FrontendOptions &fe)
{
  // don't unlink in debug mode
  if (pocl_get_bool_option("POCL_DEBUG", 0))
    return 0;

  FrontendInputFile const& file = fe.Inputs.front();
  if (file.isFile() && !file.isSystem()) {
    return pocl_remove(file.getFile().str().c_str());
  } else {
    return 0; // nothing to do
  }

}


static void get_build_log(cl_program program,
                         unsigned device_i,
                         std::stringstream &ss_build_log,
                         clang::TextDiagnosticBuffer *diagsBuffer,
                         const SourceManager &sm)
{
    for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->err_begin(),
         e = diagsBuffer->err_end(); i != e; ++i)
      {
        ss_build_log << "error: " << i->first.printToString(sm)
                     << ": " << i->second << std::endl;
      }
    for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->warn_begin(),
         e = diagsBuffer->warn_end(); i != e; ++i)
      {
        ss_build_log << "warning: " << i->first.printToString(sm)
                     << ": " << i->second << std::endl;
      }

    std::string log = ss_build_log.str();
    if (log.size() > 0) {
      POCL_MSG_ERR(log.c_str());
      pocl_cache_append_to_buildlog(program, device_i,
                                  log.c_str(),
                                  log.size());
    }
}

static llvm::Module *kernel_library(cl_device_id device);

int pocl_llvm_build_program(cl_program program,
                            unsigned device_i,
                            const char *user_options_cstr,
                            char *program_bc_path,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names,
                            int linking_program)

{
  void* write_lock = NULL;
  char tempfile[POCL_FILENAME_LENGTH];
  tempfile[0] = 0;
  llvm::Module **mod = NULL;
  char temp_include_dir[POCL_FILENAME_LENGTH];
  std::string user_options(user_options_cstr ? user_options_cstr : "");
  std::string content;
  llvm::raw_string_ostream sos(content);
  size_t n = 0;
  int error;

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  if (num_input_headers > 0) {
    error = pocl_cache_create_tempdir(temp_include_dir);
    std::string tempdir(temp_include_dir);
    assert(error == 0);

    for (n = 0; n < num_input_headers; n++) {
      char *input_header = input_headers[n]->source;
      size_t input_header_size = strlen(input_header);
      const char *header_name = header_include_names[n];
      std::string header(header_name);
      /* TODO this path stuff should be in utils */
      std::string path(tempdir);
      path.append("/");
      path.append(header_name);
      size_t last_slash = header.rfind('/');
      if (last_slash != std::string::npos) {
        std::string dir(path, 0, (tempdir.size() + 1 + last_slash));
        pocl_mkdir_p(dir.c_str());
      }
      pocl_write_file(path.c_str(), input_header, input_header_size, 0, 1);
    }
  }
  // Use CompilerInvocation::CreateFromArgs to initialize
  // CompilerInvocation. This way we can reuse the Clang's
  // command line parsing.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID =
    new clang::DiagnosticIDs();
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
    new clang::DiagnosticOptions();
  clang::TextDiagnosticBuffer *diagsBuffer =
    new clang::TextDiagnosticBuffer();

  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagsBuffer);

  CompilerInstance CI;
  CompilerInvocation &pocl_build = CI.getInvocation();

  std::stringstream ss;
  std::stringstream ss_build_log;

  // add device specific switches, if any
  // TODO this currently passes NULL as device tmpdir
  cl_device_id device = program->devices[device_i];
  if (device->ops->init_build != NULL)
    {
      char *device_switches =
        device->ops->init_build (device->data);
      if (device_switches != NULL)
        {
          ss << device_switches << " ";
        }
      POCL_MEM_FREE(device_switches);
    }

  llvm::StringRef extensions(device->extensions);

  std::string cl_ext;
  if (extensions.size() > 0) {
    size_t e_start = 0, e_end = 0;
    while (e_end < std::string::npos) {
      e_end = extensions.find(' ', e_start);
      llvm::StringRef tok = extensions.slice(e_start, e_end);
      e_start = e_end + 1;
      ss << "-D" << tok.str() << " ";
#ifndef LLVM_OLDER_THAN_4_0
      cl_ext += "+";
      cl_ext += tok.str();
      cl_ext += ",";
#endif
    }
  }
#ifndef LLVM_OLDER_THAN_4_0
  if (!cl_ext.empty()) {
    cl_ext.back() = ' '; // replace last "," with space
    ss << "-cl-ext=-all," << cl_ext;
  }
#endif
  /* temp dir takes preference */
  if (num_input_headers > 0)
    ss << "-I" << temp_include_dir << " ";

  if (device->has_64bit_long)
    ss << "-Dcl_khr_int64 ";
  // This can cause illegal optimizations when unaware
  // of the barrier semantics. -O2 is the default opt level in
  // Clang for OpenCL C and seems to affect the performance
  // of the end result, even if we optimize the final WG
  // func. TODO: There should be 'noduplicate' etc. flags in 
  // the 'barrier' function to prevent them.
  // ss << "-O2 ";

  ss << "-x cl ";
  // Remove the inline keywords to force the user functions
  // to be included in the program. Otherwise they will
  // be removed and not inlined due to -O0.
  ss << "-Dinline= ";
  // The current directory is a standard search path.
  ss << "-I. ";
  // required for clGetKernelArgInfo()
  ss << "-cl-kernel-arg-info ";

  ss << user_options << " ";

  if (device->endian_little)
    ss << "-D__ENDIAN_LITTLE__=1 ";

  if (device->image_support)
    ss << "-D__IMAGE_SUPPORT__=1 ";

  ss << "-DCL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" << device->global_var_max_size << " ";

  if (user_options.find("cl-fast-relaxed-math") != std::string::npos)
    ss << "-D__FAST_RELAXED_MATH__=1 ";

  ss << "-D__OPENCL_VERSION__=" << device->cl_version_int << " ";

  if (user_options.find("-cl-std=") == std::string::npos)
    ss << "-cl-std=" << device->cl_version_std << " ";

  std::string temp(ss.str());
  size_t pos = temp.find("-cl-std=CL");
  pos += 10;
  int cl_std_major = temp.c_str()[pos] - '0';
  int cl_std_minor = temp.c_str()[pos+2] - '0';
  int cl_std_i = cl_std_major * 100 + cl_std_minor * 10;
  ss << "-D__OPENCL_C_VERSION__=" << cl_std_i << " ";

  ss << "-fno-builtin ";
  /* with fp-contract=on we get calls to fma with processors which do not
   * have fma instructions. These ruin the performance.
   *
   * TODO find out which processors. Seems to be at least TCE
   *
   * default fp-contract is "on" which means "enable if enabled by a pragma".
   */
  llvm::Triple triple (device->llvm_target_triplet);
  if (triple.getArch () == Triple::tce)
    ss << "-ffp-contract=off ";

  // This is required otherwise the initialization fails with
  // unknown triple ''
  ss << "-triple=" << device->llvm_target_triplet << " ";
  if (device->llvm_cpu != NULL)
    ss << "-target-cpu " << device->llvm_cpu << " ";

  POCL_MSG_PRINT_LLVM("all build options: %s\n", ss.str().c_str());

  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  std::istream_iterator<std::string> i = begin;
  std::vector<const char*> itemcstrs;
  std::vector<std::string> itemstrs;
  while (i != end) 
    {
      itemstrs.push_back(*i);
      ++i;
    }

  for (unsigned idx=0; idx<itemstrs.size(); idx++)
    {
      // note: if itemstrs is modified after this, itemcstrs will be full
      // of invalid pointers! Could make copies, but would have to clean up then...
      itemcstrs.push_back(itemstrs[idx].c_str());
    }
#ifdef DEBUG_POCL_LLVM_API
  // TODO: for some reason the user_options are replicated,
  // they appear twice in a row in the output
  std::cerr << "### options: " << ss.str() 
            << "user_options: " << user_options << std::endl;
#endif

  if (program->build_log[device_i])
    POCL_MEM_FREE(program->build_log[device_i]);

  if (!CompilerInvocation::CreateFromArgs
      (pocl_build, itemcstrs.data(), itemcstrs.data() + itemcstrs.size(),
       diags)) {
    pocl_cache_create_program_cachedir(program, device_i, NULL, 0,
                                       program_bc_path);
    get_build_log(program, device_i, ss_build_log, diagsBuffer, CI.getSourceManager());
    return CL_INVALID_BUILD_OPTIONS;
  }

  LangOptions *la = pocl_build.getLangOpts();
  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();

#ifdef LLVM_OLDER_THAN_3_9
  pocl_build.setLangDefaults
    (*la, clang::IK_OpenCL, clang::LangStandard::lang_opencl12);
#else
  pocl_build.setLangDefaults
#if LLVM_OLDER_THAN_5_0
      (*la, clang::IK_OpenCL, triple, po, clang::LangStandard::lang_opencl12);
#else
      (*la, clang::InputKind::OpenCL, triple, po,
       clang::LangStandard::lang_opencl12);
#endif
#endif

  // LLVM 3.3 and older do not set that char is signed which is
  // defined by the OpenCL C specs (but not by C specs).
  la->CharIsSigned = true;

  // the per-file types don't seem to override this
  la->OpenCLVersion = cl_std_i;
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  la->FakeAddressSpaceMap = true;
#else
  la->FakeAddressSpaceMap = false;
#endif
  la->Blocks = true; //-fblocks
  la->MathErrno = false; // -fno-math-errno
  la->NoBuiltin = true;  // -fno-builtin
  la->AsmBlocks = true;  // -fasm (?)

  std::string kernelh;
  std::string BuiltinRenamesH;

  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    kernelh  = SRCDIR;
  } else {
    kernelh = PKGDATADIR;
  }
  BuiltinRenamesH = kernelh;
  kernelh += "/include/_kernel.h";
  BuiltinRenamesH += "/include/_builtin_renames.h";

  po.Includes.push_back(BuiltinRenamesH);
#ifndef LLVM_OLDER_THAN_4_0
  // Use Clang's opencl-c.h header.
  po.Includes.push_back(CLANG_RESOURCE_DIR "/include/opencl-c.h");
#endif
  po.Includes.push_back(kernelh);
  clang::TargetOptions &ta = pocl_build.getTargetOpts();
  ta.Triple = device->llvm_target_triplet;
  if (device->llvm_cpu != NULL)
    ta.CPU = device->llvm_cpu;

#ifdef DEBUG_POCL_LLVM_API
  std::cout << "### Triple: " << ta.Triple.c_str() <<  ", CPU: " << ta.CPU.c_str();
#endif
  CI.createDiagnostics(diagsBuffer, false);

  FrontendOptions &fe = pocl_build.getFrontendOpts();
  // The CreateFromArgs created an stdin input which we should remove first.
  fe.Inputs.clear();
  if (load_source(fe, program) != 0)
    return CL_OUT_OF_HOST_MEMORY;

  CodeGenOptions &cg = pocl_build.getCodeGenOpts();
  cg.EmitOpenCLArgMetadata = true;
  cg.StackRealignment = true;
  // Let the vectorizer or another optimization pass unroll the loops,
  // in case it sees beneficial.
  cg.UnrollLoops = false;
  // Lets leave vectorization to later compilation phase
  cg.VectorizeLoop = false;
  cg.VectorizeSLP = false;
  // This workarounds a Frontend codegen issues with an illegal address
  // space cast which is later flattened (and thus implicitly fixed) in
  // the TargetAddressSpaces. See:  https://github.com/pocl/pocl/issues/195
  cg.VerifyModule = false;

  PreprocessorOutputOptions &poo = pocl_build.getPreprocessorOutputOpts();
  poo.ShowCPP = 1;
  poo.ShowComments = 0;
  poo.ShowLineMarkers = 0;
  poo.ShowMacroComments = 0;
  poo.ShowMacros = 1;
  poo.RewriteIncludes = 0;

  pocl_cache_tempname(tempfile, ".cl", NULL);
  fe.OutputFile.assign(tempfile);

  bool success = true;
  clang::PrintPreprocessedAction Preprocess;
  success = CI.ExecuteAction(Preprocess);
  char *PreprocessedOut = nullptr;
  uint64_t PreprocessedSize = 0;

  if (success) {
    pocl_read_file(tempfile, &PreprocessedOut, &PreprocessedSize);
  }
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0) {
    if (num_input_headers > 0)
      pocl_rm_rf(temp_include_dir);
    pocl_remove(tempfile);
  }

  if (PreprocessedOut == nullptr) {
    pocl_cache_create_program_cachedir(program, device_i, NULL, 0,
                                       program_bc_path);
    get_build_log(program, device_i, ss_build_log, diagsBuffer, CI.getSourceManager());
    return CL_BUILD_PROGRAM_FAILURE;
  }

  pocl_cache_create_program_cachedir(program, device_i, PreprocessedOut,
                                     static_cast<size_t>(PreprocessedSize), program_bc_path);

  POCL_MEM_FREE(PreprocessedOut);

  if (pocl_exists(program_bc_path)) {
    unlink_source(fe);
    return CL_SUCCESS;
  }

  // TODO: use pch: it is possible to disable the strict checking for
  // the compilation flags used to compile it and the current translation
  // unit via the preprocessor options directly.
  clang::EmitLLVMOnlyAction EmitLLVM(GlobalContext());
  success = CI.ExecuteAction(EmitLLVM);

  unlink_source(fe);

  get_build_log(program, device_i, ss_build_log, diagsBuffer, CI.getSourceManager());

  if (!success)
    return CL_BUILD_PROGRAM_FAILURE;

  mod = (llvm::Module **)&program->llvm_irs[device_i];
  if (*mod != NULL) {
    delete *mod;
    --numberOfIRs;
  }

  *mod = EmitLLVM.takeModule().release();

  if (*mod == NULL)
    return CL_BUILD_PROGRAM_FAILURE;

  ++numberOfIRs;

  // link w kernel lib, but not if we're called from clCompileProgram()
  // Later this should be replaced with indexed linking of source code
  // and/or bitcode for each kernel.
  if (linking_program) {
    currentPoclDevice = device;
    llvm::Module *libmodule = kernel_library(device);
    assert(libmodule != NULL);
    std::string log("Error(s) while linking: \n");
    if (link(*mod, libmodule, log)) {
      POCL_MSG_ERR(log.c_str());
      pocl_cache_append_to_buildlog(program, device_i, log.c_str(), log.size());
      delete *mod;
      *mod = nullptr;
      --numberOfIRs;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  write_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
  assert(write_lock);

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc. Its required in clBuildProgram */
  error = pocl_write_module(*mod, program_bc_path, 0);
  assert(error == 0);

  /* To avoid writing & reading the same back,
   * save program->binaries[i]
   */
  WriteBitcodeToFile(*mod, sos);
  sos.str(); // flush

  if (program->binaries[device_i])
    POCL_MEM_FREE(program->binaries[device_i]);

  n = content.size();
  program->binary_sizes[device_i] = n;
  program->binaries[device_i] = (unsigned char *) malloc(n);
  std::memcpy(program->binaries[device_i], content.c_str(), n);

  pocl_cache_release_lock(write_lock);

  return CL_SUCCESS;
}


int pocl_llvm_link_program(cl_program program,
                           unsigned device_i,
                           char *program_bc_path,
                           cl_uint num_input_programs,
                           unsigned char **cur_device_binaries,
                           size_t *cur_device_binary_sizes,
                           void **cur_llvm_irs) {

  void *write_lock;
  std::string content;
  std::string concated_binaries;
  llvm::raw_string_ostream sos(content);
  size_t n = 0, i;
  cl_device_id device = program->devices[device_i];
  llvm::Module **modptr = (llvm::Module **)&program->llvm_irs[device_i];
  int error;

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  LLVMContext *c = GlobalContext();
#ifdef LLVM_OLDER_THAN_3_8
  llvm::Module *mod = new llvm::Module(StringRef("linked_program"), *c);
#else
  std::unique_ptr<llvm::Module> mod(
      new llvm::Module(StringRef("linked_program"), *c));
#endif

  for (i = 0; i < num_input_programs; i++) {
    assert(cur_device_binaries[i]);
    assert(cur_device_binary_sizes[i]);
    concated_binaries.append(std::string((char *)cur_device_binaries[i],
                                         cur_device_binary_sizes[i]));

    llvm::Module *p = (llvm::Module *)cur_llvm_irs[i];
    assert(p);

#ifdef LLVM_OLDER_THAN_3_8
    if (Linker::LinkModules(mod, llvm::CloneModule(p))) {
      delete mod;
      return CL_LINK_PROGRAM_FAILURE;
    }
#else
    if (Linker::linkModules(*mod, llvm::CloneModule(p))) {
      return CL_LINK_PROGRAM_FAILURE;
    }
#endif
  }

#ifdef LLVM_OLDER_THAN_3_8
  llvm::Module *linked_module = mod;
#else
  llvm::Module *linked_module = mod.release();
#endif

  if (linked_module == nullptr)
    return CL_LINK_PROGRAM_FAILURE;

  if (*modptr != nullptr) {
    delete *modptr;
    --numberOfIRs;
  }
  *modptr = nullptr;

  // linked all the programs together, now link in the kernel library
  currentPoclDevice = device;
  llvm::Module *libmodule = kernel_library(device);
  assert(libmodule != NULL);
  std::string log("Error(s) while linking: \n");
  if (link(linked_module, libmodule, log)) {
    POCL_MSG_ERR(log.c_str());
    pocl_cache_append_to_buildlog(program, device_i, log.c_str(), log.size());
    delete linked_module;
    return CL_BUILD_PROGRAM_FAILURE;
  }

  *modptr = linked_module;
  ++numberOfIRs;

  /* TODO currently cached on concated binary contents (in undefined order),
     this is not terribly useful (but we have to store it somewhere..) */
  error = pocl_cache_create_program_cachedir(program, device_i,
                                     concated_binaries.c_str(),
                                     concated_binaries.size(),
                                     program_bc_path);
  assert(error == 0);


  write_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
  assert(write_lock);

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc. Its required in clBuildProgram */
  error = pocl_write_module(linked_module, program_bc_path, 0);
  assert(error == 0);

  /* To avoid writing & reading the same back,
   * save program->binaries[i]
   */
  WriteBitcodeToFile(linked_module, sos);
  sos.str(); // flush

  if (program->binaries[device_i])
    POCL_MEM_FREE(program->binaries[device_i]);

  n = content.size();
  program->binary_sizes[device_i] = n;
  program->binaries[device_i] = (unsigned char *)malloc(n);
  std::memcpy(program->binaries[device_i], content.c_str(), n);

  pocl_cache_release_lock(write_lock);

  return CL_SUCCESS;
}

// The old way of getting kernel metadata from "opencl.kernels" module meta.
// LLVM < 3.9 and SPIR
static int pocl_get_kernel_arg_module_metadata(const char* kernel_name,
                                               llvm::Module *input,
                                               cl_kernel kernel)
{
  // find the right kernel in "opencl.kernels" metadata
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  llvm::MDNode *kernel_metadata = NULL;

  assert(opencl_kernels && opencl_kernels->getNumOperands());

  for (unsigned i = 0, e = opencl_kernels->getNumOperands(); i != e; ++i) {
    llvm::MDNode *kernel_iter = opencl_kernels->getOperand(i);

    llvm::Value *meta =
      dyn_cast<llvm::ValueAsMetadata>(kernel_iter->getOperand(0))->getValue();
    llvm::Function *kernel_prototype = llvm::cast<llvm::Function>(meta);
    std::string name = kernel_prototype->getName().str();
    if (name == kernel_name) {
      kernel_metadata = kernel_iter;
      break;
    }
  }

  kernel->arg_info =
    (struct pocl_argument_info*)calloc(
      kernel->num_args, sizeof(struct pocl_argument_info));
  memset(
    kernel->arg_info, 0, sizeof(struct pocl_argument_info) * kernel->num_args);

  kernel->has_arg_metadata = 0;

  assert(kernel_metadata && "kernel NOT found in opencl.kernels metadata");

#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  int BitcodeIsSPIR = input->getTargetTriple().find("spir") == 0;
#endif

  unsigned e = kernel_metadata->getNumOperands();
  for (unsigned i = 1; i != e; ++i) {
    llvm::MDNode *meta_node =
      llvm::cast<MDNode>(kernel_metadata->getOperand(i));

    // argument num
    unsigned arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
    int has_meta_for_every_arg = ((arg_num-1) == kernel->num_args);
#endif

    llvm::MDString *meta_name_node = llvm::cast<MDString>(meta_node->getOperand(0));
    std::string meta_name = meta_name_node->getString().str();

    for (unsigned j = 1; j != arg_num; ++j) {
      llvm::Value *meta_arg_value = NULL;
      if (isa<ValueAsMetadata>(meta_node->getOperand(j)))
        meta_arg_value =
          dyn_cast<ValueAsMetadata>(meta_node->getOperand(j))->getValue();
      else if (isa<ConstantAsMetadata>(meta_node->getOperand(j)))
        meta_arg_value =
          dyn_cast<ConstantAsMetadata>(meta_node->getOperand(j))->getValue();
      struct pocl_argument_info* current_arg = &kernel->arg_info[j-1];

      if (meta_arg_value != NULL && isa<ConstantInt>(meta_arg_value) &&
          meta_name == "kernel_arg_addr_space") {
        assert(has_meta_for_every_arg && "kernel_arg_addr_space meta incomplete");
        kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
        //std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
        llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
        uint64_t val = m->getLimitedValue(UINT_MAX);
        bool SPIRAddressSpaceIDs;
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
        SPIRAddressSpaceIDs = BitcodeIsSPIR;
#else
        // We have an LLVM fixed to produce always SPIR AS ids for the argument
        // info metadata.
        SPIRAddressSpaceIDs = true;
#endif

        if (SPIRAddressSpaceIDs) {
          switch(val) {
            case 0:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
            case 1:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL; break;
            case 3:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL; break;
            case 2:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT; break;
          }
        } else {
          switch(val) {
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
            case POCL_FAKE_AS_PRIVATE:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
            case POCL_FAKE_AS_GLOBAL:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL; break;
            case POCL_FAKE_AS_LOCAL:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL; break;
            case POCL_FAKE_AS_CONSTANT:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT; break;
            case POCL_FAKE_AS_GENERIC:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
#endif
          default:
            POCL_MSG_ERR("Unknown address space ID %lu\n", val);
            break;
          }
        }
      }
      else if (isa<MDString>(meta_node->getOperand(j))) {
        //std::cout << "is MDString" << std::endl;
        llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
        std::string val = m->getString().str();

        if (meta_name == "kernel_arg_access_qual") {
          assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
          if (val == "read_write")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
          else if (val == "read_only")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_ONLY;
          else if (val == "write_only")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
          else if (val == "none")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
          else
            std::cout << "UNKNOWN kernel_arg_access_qual value: " << val << std::endl;
        } else if (meta_name == "kernel_arg_type") {
          assert(has_meta_for_every_arg && "kernel_arg_type meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_NAME;
          current_arg->type_name = (char *)malloc (val.size () + 1);
          std::strcpy(current_arg->type_name, val.c_str());
        } else if (meta_name == "kernel_arg_base_type") {
          // may or may not be present even in SPIR
        } else if (meta_name == "kernel_arg_type_qual") {
          assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER;
          current_arg->type_qualifier = 0;
          if (val.find("const") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
          if (val.find("restrict") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
          if (val.find("volatile") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
        } else if (meta_name == "kernel_arg_name") {
          assert(has_meta_for_every_arg && "kernel_arg_name meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
          current_arg->name = (char *)malloc (val.size () + 1);
          std::strcpy(current_arg->name, val.c_str());
        } else
          std::cout << "UNKNOWN opencl metadata name: " << meta_name << std::endl;
      }
      else if (meta_name != "reqd_work_group_size")
        std::cout << "UNKNOWN opencl metadata class for: " << meta_name << std::endl;

    }
  }
  return 0;
}

static std::map<std::string, unsigned> type_size_map = {
  {std::string("char"), (1)},
  {std::string("uchar"), (1)},
  {std::string("short"), (2)},
  {std::string("ushort"), (2)},
  {std::string("int"), (4)},
  {std::string("uint"), (4)},
  {std::string("long"), (8)},
  {std::string("ulong"), (8)},


  {std::string("char2"), (1*2)},
  {std::string("uchar2"), (1*2)},
  {std::string("short2"), (2*2)},
  {std::string("ushort2"), (2*2)},
  {std::string("int2"), (4*2)},
  {std::string("uint2"), (4*2)},
  {std::string("long2"), (8*2)},
  {std::string("ulong2"), (8*2)},


  {std::string("char3"), (1*4)},
  {std::string("uchar3"), (1*4)},
  {std::string("short3"), (2*4)},
  {std::string("ushort3"), (2*4)},
  {std::string("int3"), (4*4)},
  {std::string("uint3"), (4*4)},
  {std::string("long3"), (8*4)},
  {std::string("ulong3"), (8*4)},


  {std::string("char4"), (1*4)},
  {std::string("uchar4"), (1*4)},
  {std::string("short4"), (2*4)},
  {std::string("ushort4"), (2*4)},
  {std::string("int4"), (4*4)},
  {std::string("uint4"), (4*4)},
  {std::string("long4"), (8*4)},
  {std::string("ulong4"), (8*4)},


  {std::string("char8"), (1*8)},
  {std::string("uchar8"), (1*8)},
  {std::string("short8"), (2*8)},
  {std::string("ushort8"), (2*8)},
  {std::string("int8"), (4*8)},
  {std::string("uint8"), (4*8)},
  {std::string("long8"), (8*8)},
  {std::string("ulong8"), (8*8)},

  {std::string("char16"), (1*16)},
  {std::string("uchar16"), (1*16)},
  {std::string("short16"), (2*16)},
  {std::string("ushort16"), (2*16)},
  {std::string("int16"), (4*16)},
  {std::string("uint16"), (4*16)},
  {std::string("long16"), (8*16)},
  {std::string("ulong16"), (8*16)},

  {std::string("half"), (2)},
  {std::string("float"), (4)},
  {std::string("double"), (8)},

  {std::string("half2"), (2*2)},
  {std::string("float2"), (4*2)},
  {std::string("double2"), (8*2)},

  {std::string("half3"), (2*4)},
  {std::string("float3"), (4*4)},
  {std::string("double3"), (8*4)},

  {std::string("half4"), (2*4)},
  {std::string("float4"), (4*4)},
  {std::string("double4"), (8*4)},

  {std::string("half8"), (2*8)},
  {std::string("float8"), (4*8)},
  {std::string("double8"), (8*8)},

  {std::string("half16"), (2*16)},
  {std::string("float16"), (4*16)},
  {std::string("double16"), (8*16)}
};

#ifndef LLVM_OLDER_THAN_3_9
// Clang 3.9 uses function metadata instead of module metadata for presenting
// OpenCL kernel information.
static int pocl_get_kernel_arg_function_metadata(const char* kernel_name,
                                                 llvm::Module *input,
                                                 cl_kernel kernel)
{
  llvm::Function *Kernel = NULL;
  int bitcode_is_spir = input->getTargetTriple().find("spir") == 0;

  // SPIR still uses the "opencl.kernels" MD.
  if(bitcode_is_spir)
    return pocl_get_kernel_arg_module_metadata(kernel_name, input, kernel);

  for (llvm::Module::iterator i = input->begin(), e = input->end();
       i != e; ++i) {
    if (i->getMetadata("kernel_arg_access_qual")
        && i->getName() == kernel_name)
      {
        Kernel = &*i;
        break;
      }
  }
  assert(Kernel);
  kernel->has_arg_metadata = 0;

  llvm::MDNode *meta_node;
  llvm::Value *meta_arg_value = NULL;
  struct pocl_argument_info* current_arg = NULL;

  kernel->arg_info =
    (struct pocl_argument_info*)calloc(
      kernel->num_args, sizeof(struct pocl_argument_info));
  memset(
    kernel->arg_info, 0, sizeof(struct pocl_argument_info) * kernel->num_args);

  // kernel_arg_addr_space
  meta_node = Kernel->getMetadata("kernel_arg_addr_space");
  assert(meta_node != nullptr);
  unsigned arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  int has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  for (unsigned j = 0; j < arg_num; ++j) {
    assert(has_meta_for_every_arg && "kernel_arg_addr_space meta incomplete");

    current_arg = &kernel->arg_info[j];
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
    //std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
     meta_arg_value =
          dyn_cast<ConstantAsMetadata>(meta_node->getOperand(j))->getValue();
    llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
    uint64_t val = m->getLimitedValue(UINT_MAX);

    bool SPIRAddressSpaceIDs;
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
    SPIRAddressSpaceIDs = bitcode_is_spir;
#else
    // We have an LLVM fixed to produce always SPIR AS ids for the argument
    // info metadata.
    SPIRAddressSpaceIDs = true;
#endif
    if (SPIRAddressSpaceIDs) {
      switch(val) {
      case 0:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
      case 1:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL; break;
      case 3:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL; break;
      case 2:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT; break;
      default:
        POCL_MSG_ERR("Unknown address space ID %lu\n", val);
        break;
      }
    } else {
      switch(val) {
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
      case POCL_FAKE_AS_PRIVATE:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
      case POCL_FAKE_AS_GLOBAL:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL; break;
      case POCL_FAKE_AS_LOCAL:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL; break;
      case POCL_FAKE_AS_CONSTANT:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT; break;
      case POCL_FAKE_AS_GENERIC:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
#endif
      default:
        POCL_MSG_ERR("Unknown address space ID %lu\n", val);
        break;
      }
    }
  }

  // kernel_arg_access_qual
  meta_node = Kernel->getMetadata("kernel_arg_access_qual");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");

  for (unsigned j= 0; j < meta_node->getNumOperands(); ++j) {
    current_arg = &kernel->arg_info[j];
    //std::cout << "is MDString" << std::endl;
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
    if (val == "read_write")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
    else if (val == "read_only")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_ONLY;
    else if (val == "write_only")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
    else if (val == "none")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
    else
      std::cout << "UNKNOWN kernel_arg_access_qual value: " << val << std::endl;
  }

  // kernel_arg_type
  meta_node = Kernel->getMetadata("kernel_arg_type");
  assert(meta_node != nullptr);
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_type meta incomplete");

  for (unsigned j= 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel->arg_info[j];
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_NAME;
    current_arg->type_name = (char *)malloc (val.size () + 1);
    if (type_size_map.find(val) != type_size_map.end())
      current_arg->type_size = type_size_map[val];
    else
      current_arg->type_size = 0;
    std::strcpy(current_arg->type_name, val.c_str());
  }

  // kernel_arg_type_qual
  meta_node = Kernel->getMetadata("kernel_arg_type_qual");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
  for (unsigned j= 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel->arg_info[j];
    assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER;
    current_arg->type_qualifier = 0;
    if (val.find("const") != std::string::npos)
      current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
    if (val.find("restrict") != std::string::npos)
      current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
    if (val.find("volatile") != std::string::npos)
      current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
  }

  //kernel_arg_name
  meta_node = Kernel->getMetadata("kernel_arg_name");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_name meta incomplete");
  for (unsigned j= 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel->arg_info[j];
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
    current_arg->name = (char *)malloc (val.size () + 1);
    std::strcpy(current_arg->name, val.c_str());
  }

  return 0;
}
#endif

int pocl_llvm_get_kernel_metadata(cl_program program,
                                  cl_kernel kernel,
                                  int device_i,
                                  const char* kernel_name,
                                  int * errcode)
{

  int i;
  llvm::Module *input = NULL;
  cl_device_id Device = program->devices[device_i];

  assert(Device->llvm_target_triplet &&
         "Device has no target triple set");

  if (program->llvm_irs != NULL &&
      program->llvm_irs[device_i] != NULL)
    input = (llvm::Module*)program->llvm_irs[device_i];
  else {
    *errcode = CL_INVALID_PROGRAM_EXECUTABLE;
    return 1;
  }

  llvm::Function *KernelFunction = input->getFunction(kernel_name);
  if (!KernelFunction) {
    *errcode = CL_INVALID_KERNEL_NAME;
    return 1;
  }
  kernel->num_args = KernelFunction->arg_size();

#if defined(LLVM_OLDER_THAN_3_9)
  if (pocl_get_kernel_arg_module_metadata(kernel_name, input, kernel)) {
    *errcode = CL_INVALID_KERNEL;
    return 1;
  }
#else
  if (pocl_get_kernel_arg_function_metadata(kernel_name, input, kernel)) {
    *errcode = CL_INVALID_KERNEL;
    return 1;
  }
#endif

#ifdef DEBUG_POCL_LLVM_API
  printf("### fetching kernel metadata for kernel %s program %p input llvm::Module %p\n",
         kernel_name, program, input);
#endif

  DataLayout *TD = nullptr;
#ifdef LLVM_OLDER_THAN_3_7
  const std::string &ModuleDataLayout =
    input->getDataLayout()->getStringRepresentation();
#else
  const std::string &ModuleDataLayout =
    input->getDataLayout().getStringRepresentation();
#endif
  assert (!ModuleDataLayout.empty ());
  TD = new DataLayout (ModuleDataLayout);

  SmallVector<GlobalVariable *, 8> locals;
  for (llvm::Module::global_iterator i = input->global_begin(),
         e = input->global_end();
       i != e; ++i) {
    std::string funcName = "";
    funcName = KernelFunction->getName().str();
    if (pocl::isAutomaticLocal(funcName, *i)) {
      POCL_MSG_PRINT_LLVM("Automatic local detected: %s\n",
                          i->getName().str().c_str());
      locals.push_back(&*i);
    }
  }

  kernel->num_locals = locals.size();

  /* Temporary store for the arguments that are set with clSetKernelArg. */
  kernel->dyn_arguments =
    (struct pocl_argument *) malloc ((kernel->num_args + kernel->num_locals) *
                                     sizeof (struct pocl_argument));
  /* Initialize kernel "dynamic" arguments (in case the user doesn't). */
  for (unsigned i = 0; i < kernel->num_args; ++i)
    {
      kernel->dyn_arguments[i].value = NULL;
      kernel->dyn_arguments[i].size = 0;
    }

  /* Fill up automatic local arguments. */
  for (unsigned i = 0; i < kernel->num_locals; ++i)
    {
      unsigned auto_local_size =
        TD->getTypeAllocSize(locals[i]->getInitializer()->getType());
      kernel->dyn_arguments[kernel->num_args + i].value = NULL;
      kernel->dyn_arguments[kernel->num_args + i].size = auto_local_size;
#ifdef DEBUG_POCL_LLVM_API
      printf("### automatic local %d size %u\n", i, auto_local_size);
#endif
    }

  i = 0;
  for (llvm::Function::const_arg_iterator ii = KernelFunction->arg_begin(),
                                          ee = KernelFunction->arg_end();
       ii != ee; ii++) {
    llvm::Type *t = ii->getType();
    struct pocl_argument_info &ArgInfo = kernel->arg_info[i];
    ArgInfo.type = POCL_ARG_TYPE_NONE;
    ArgInfo.is_local = false;
    const llvm::PointerType *p = dyn_cast<llvm::PointerType>(t);
    if (p && !ii->hasByValAttr()) {
      ArgInfo.type = POCL_ARG_TYPE_POINTER;
      // index 0 is for function attributes, parameters start at 1.
      // TODO: detect the address space from MD.

#ifndef POCL_USE_FAKE_ADDR_SPACE_IDS
      if (ArgInfo.address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
        ArgInfo.is_local = true;
#else
      if (p->getAddressSpace() == POCL_FAKE_AS_GLOBAL ||
          p->getAddressSpace() == POCL_FAKE_AS_CONSTANT ||
          pocl::is_image_type(*t) || pocl::is_sampler_type(*t))
        {
          kernel->arg_info[i].is_local = false;
        }
      else
        {
          if (p->getAddressSpace() != POCL_FAKE_AS_LOCAL)
            {
              p->dump();
              assert(p->getAddressSpace() == POCL_FAKE_AS_LOCAL);
            }
          kernel->arg_info[i].is_local = true;
        }
#endif
    }

    if (pocl::is_image_type(*t)) {
      ArgInfo.type = POCL_ARG_TYPE_IMAGE;
    } else if (pocl::is_sampler_type(*t)) {
      ArgInfo.type = POCL_ARG_TYPE_SAMPLER;
    }
    i++;
  }
  // fill 'kernel->reqd_wg_size'
  kernel->reqd_wg_size = (size_t *)malloc(3 * sizeof(size_t));

  size_t reqdx = 0, reqdy = 0, reqdz = 0;

#ifdef LLVM_OLDER_THAN_3_9
  llvm::NamedMDNode *size_info =
    KernelFunction->getParent()->getNamedMetadata("opencl.kernel_wg_size_info");
  if (size_info) {
    for (unsigned i = 0, e = size_info->getNumOperands(); i != e; ++i) {
      llvm::MDNode *KernelSizeInfo = size_info->getOperand(i);
      if (dyn_cast<ValueAsMetadata>(
        KernelSizeInfo->getOperand(0).get())->getValue() != KernelFunction)
        continue;
      reqdx = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   KernelSizeInfo->getOperand(1))->getValue()))->getLimitedValue();
      reqdy = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   KernelSizeInfo->getOperand(2))->getValue()))->getLimitedValue();
      reqdz = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   KernelSizeInfo->getOperand(3))->getValue()))->getLimitedValue();
      break;
    }
  }
#else
  llvm::MDNode *ReqdWGSize =
    KernelFunction->getMetadata("reqd_work_group_size");
  if (ReqdWGSize != NULL) {
    reqdx = (llvm::cast<ConstantInt>(
               llvm::dyn_cast<ConstantAsMetadata>(
                 ReqdWGSize->getOperand(0))->getValue()))->getLimitedValue();
    reqdy = (llvm::cast<ConstantInt>(
               llvm::dyn_cast<ConstantAsMetadata>(
                 ReqdWGSize->getOperand(1))->getValue()))->getLimitedValue();
    reqdz = (llvm::cast<ConstantInt>(
               llvm::dyn_cast<ConstantAsMetadata>(
                 ReqdWGSize->getOperand(2))->getValue()))->getLimitedValue();
  }
#endif

  kernel->reqd_wg_size[0] = reqdx;
  kernel->reqd_wg_size[1] = reqdy;
  kernel->reqd_wg_size[2] = reqdz;

#ifndef POCL_ANDROID
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous standalone devices which
  // need the definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now

  std::stringstream content;

  content << std::endl << "#include <pocl_device.h>" << std::endl
          << "void _pocl_launcher_" << kernel_name
          << "_workgroup(void** args, struct pocl_context*);" << std::endl
          << "void _pocl_launcher_" << kernel_name
          << "_workgroup_fast(void** args, struct pocl_context*);" << std::endl;

  if (Device->global_as_id != 0)
    content << "__attribute__((address_space(" << Device->global_as_id << ")))"
            << std::endl;

  content << "__kernel_metadata _" << kernel_name << "_md = {" << std::endl
          << "     \"" << kernel_name << "\"," << std::endl
          << "     " << kernel->num_args << "," << std::endl
          << "     " << kernel->num_locals << "," << std::endl
          << "     _pocl_launcher_" << kernel_name << "_workgroup_fast" << std::endl
          << " };" << std::endl;

  pocl_cache_write_descriptor(program, device_i,
                              kernel_name, content.str().c_str(),
                              content.str().size());
#endif

  delete TD;
  *errcode = CL_SUCCESS;
  return 0;
}

char* get_cpu_name() {
#ifdef __mips__
  // The MIPS backend isn't able to automatically detect the host yet and the
  // value returned by llvm::sys::getHostCPUName() isn't usable in the
  // -target-cpu option so we must use the CPU detected by CMake.
  StringRef r = OCL_KERNEL_TARGET_CPU;
#else
  StringRef r = llvm::sys::getHostCPUName();
#endif

#ifdef LLVM_3_8
  // https://github.com/pocl/pocl/issues/413
  if (r.str() == "skylake") {
    r = llvm::StringRef("haswell");
  }
#endif

  if (r.str() == "generic") {
    POCL_MSG_WARN("LLVM does not recognize your cpu, trying to use "
                   HOST_CPU " for -target-cpu\n");
    r = llvm::StringRef(HOST_CPU);
  }

  assert(r.size() > 0);
  char* cpu_name = (char*) malloc (r.size()+1);
  strncpy(cpu_name, r.data(), r.size());
  cpu_name[r.size()] = 0;
  return cpu_name;
}

int cpu_has_fma() {
  StringMap<bool> features;
  bool res = llvm::sys::getHostCPUFeatures(features);
  assert(res);
  return ((features["fma"] || features["fma4"]) ? 1 : 0);
}

/* helpers copied from LLVM opt START */

/* FIXME: these options should come from the cl_device, and
 * cl_program's options. */
static llvm::TargetOptions GetTargetOptions() {
  llvm::TargetOptions Options;
#ifdef LLVM_OLDER_THAN_3_9
  Options.PositionIndependentExecutable = true;
#endif
  #ifdef HOST_FLOAT_SOFT_ABI
  Options.FloatABIType = FloatABI::Soft;
  #else
  Options.FloatABIType = FloatABI::Hard;
  #endif
#if 0
  Options.LessPreciseFPMADOption = EnableFPMAD;
  Options.NoFramePointerElim = DisableFPElim;
  Options.NoFramePointerElimNonLeaf = DisableFPElimNonLeaf;
  Options.AllowFPOpFusion = FuseFPOps;
  Options.UnsafeFPMath = EnableUnsafeFPMath;
  Options.NoInfsFPMath = EnableNoInfsFPMath;
  Options.NoNaNsFPMath = EnableNoNaNsFPMath;
  Options.HonorSignDependentRoundingFPMathOption =
  EnableHonorSignDependentRoundingFPMath;
  Options.UseSoftFloat = GenerateSoftFloatCalls;
  if (FloatABIForCalls != FloatABI::Default)
    Options.FloatABIType = FloatABIForCalls;
  Options.NoZerosInBSS = DontPlaceZerosInBSS;
  Options.GuaranteedTailCallOpt = EnableGuaranteedTailCallOpt;
  Options.DisableTailCalls = DisableTailCalls;
  Options.StackAlignmentOverride = OverrideStackAlignment;
  Options.RealignStack = EnableRealignStack;
  Options.TrapFuncName = TrapFuncName;
  Options.EnableSegmentedStacks = SegmentedStacks;
  Options.UseInitArray = UseInitArray;
  Options.SSPBufferSize = SSPBufferSize;
#endif
  return Options;
}

/* for "distro" style kernel libs, return which kernellib to use, at runtime */
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
static const char* getX86KernelLibName() {
  StringMap<bool> Features;
  llvm::sys::getHostCPUFeatures(Features);
  const char *res = NULL;

  if (Features["sse2"])
    res = "sse2";
  else
    POCL_ABORT("Pocl on x86_64 requires at least SSE2");
  if (Features["ssse3"] && Features["cx16"])
    res = "ssse3";
  if (Features["sse4.1"] && Features["cx16"])
    res = "sse41";
  if (Features["avx"] && Features["cx16"] && Features["popcnt"])
    res = "avx";
  if (Features["avx"] && Features["cx16"] && Features["popcnt"]
      && Features["xop"] && Features["fma4"])
    res = "avx_fma4";
  if (Features["avx"] && Features["avx2"] && Features["cx16"]
      && Features["popcnt"] && Features["lzcnt"] && Features["f16c"]
      && Features["fma"] && Features["bmi"] && Features["bmi2"])
    res = "avx2";
  if (Features["avx512f"] )
    res = "avx512";

  return res;
}
#endif

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(cl_device_id device) {

  if (targetMachines.find(device) != targetMachines.end())
    return targetMachines[device];

  std::string Error;
  Triple TheTriple(device->llvm_target_triplet);

  std::string MCPU = device->llvm_cpu ? device->llvm_cpu : "";

  const Target *TheTarget = 
    TargetRegistry::lookupTarget("", TheTriple, Error);
  
  // In LLVM 3.4 and earlier, the target registry falls back to 
  // the cpp backend in case a proper match was not found. In 
  // that case simply do not use target info in the compilation 
  // because it can be an off-tree target not registered at
  // this point (read: TCE).
  if (!TheTarget || TheTarget->getName() == std::string("cpp")) {
    return 0;
  }
  
  TargetMachine* TM = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                                     MCPU, StringRef(""),
                                                     GetTargetOptions(),
                                                     Reloc::PIC_, 
                                                     CodeModel::Default,
                                                     CodeGenOpt::Aggressive);
  assert (TM != NULL && "llvm target has no targetMachine constructor"); 
  if (device->ops->init_target_machine)
    device->ops->init_target_machine(device->data, TM);
  targetMachines[device] = TM;

  return TM;
}
/* helpers copied from LLVM opt END */


/* Check if a module references read_image{i,ui,f} or write_image{i,ui,f}
 * from pocl kernel library */
static bool moduleReferencesImages(llvm::Module *M) {
  bool refs_images = false;
  /* all of pocl's read/write pix functions use these internally */
  const llvm::StringRef readpix("pocl_read_pixel");
  const llvm::StringRef writepix("pocl_write_pixel");

  for (llvm::Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    llvm::Function *f = &*i;
    if (f->getName().equals(readpix) || f->getName().equals(writepix)) {
      refs_images = true;
      break;
    }
  }

  return refs_images;
}

/**
 * Prepare the kernel compiler passes.
 *
 * The passes are created only once per program run per device.
 * The returned pass manager should not be modified, only the Module
 * should be optimized using it.
 */
static PassManager &
kernel_compiler_passes(cl_device_id device, llvm::Module *input,
                       const std::string &module_data_layout) {

  PassManager *Passes = nullptr;
  PassRegistry *Registry = nullptr;

  bool refs_images = moduleReferencesImages(input);

  std::map<cl_device_id, PassManager *> &kernel_pass_map =
      (refs_images ? kernelPassesNoflatten : kernelPassesFlatten);

  if (kernel_pass_map.find(device) != kernel_pass_map.end()) {
    return *kernel_pass_map[device];
    }

    bool SPMDDevice = device->spmd;

    Registry = PassRegistry::getPassRegistry();

    Triple triple(device->llvm_target_triplet);

    Passes = new PassManager();

    // Need to setup the target info for target specific passes. */
    TargetMachine *Machine = GetTargetMachine(device);

#ifdef LLVM_OLDER_THAN_3_7
    // Add internal analysis passes from the target machine.
    if (Machine)
      Machine->addAnalysisPasses(*Passes);
#else
    if (Machine)
      Passes->add(
          createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));
#endif

  if (module_data_layout != "") {
#if (defined LLVM_OLDER_THAN_3_7)
    Passes->add(new DataLayoutPass());
#endif
  }

  /* Disables automated generation of libcalls from code patterns. 
     TCE doesn't have a runtime linker which could link the libs later on.
     Also the libcalls might be harmful for WG autovectorization where we 
     want to try to vectorize the code it converts to e.g. a memset or 
     a memcpy */
#ifdef LLVM_OLDER_THAN_3_7
  TargetLibraryInfo *TLI = new TargetLibraryInfo(triple);
  TLI->disableAllFunctions();
  Passes->add(TLI);
#else
  TargetLibraryInfoImpl TLII(triple);
  TLII.disableAllFunctions();
  Passes->add(new TargetLibraryInfoWrapperPass(TLII));
#endif

  /* The kernel compiler passes to run, in order.

     Notes about the kernel compiler phase ordering:
     -mem2reg first because we get unoptimized output from Clang where all
     variables are allocas. Avoid context saving the allocas and make the
     more readable by calling -mem2reg at the beginning.

     -implicit-cond-barriers after -implicit-loop-barriers because the latter can inject
     barriers to loops inside conditional regions after which the peeling should be 
     avoided by injecting the implicit conditional barriers

     -loop-barriers, -barriertails, and -barriers should be ran after the implicit barrier 
     injection passes so they "normalize" the implicit barriers also

     -phistoallocas before -workitemloops as otherwise it cannot inject context
     restore code (PHIs need to be at the beginning of the BB and so one cannot
     context restore them with non-PHI code if the value is needed in another PHI). */

  std::vector<std::string> passes;
  passes.push_back("remove-optnone");
  passes.push_back("optimize-wi-func-calls");
  passes.push_back("handle-samplers");
  passes.push_back("workitem-handler-chooser");
  passes.push_back("mem2reg");
  passes.push_back("domtree");
  if (device->autolocals_to_args)
	  passes.push_back("automatic-locals");

  if (SPMDDevice)
    passes.push_back("flatten-inline-all");
  else {
    /* LLVM is either still not intelligent enough to flatten,
     * or we're missing some pass.
     *
     * The only reason we've reintroduced "global-only" flatten was that
     * read/write image functions are very "heavy" and a conformance test
     * that tests a kernel with 127 image arguments, would flatten 127
     * write_image() calls, which compiled down to a several megabyte IR
     *
     * read/write images are the only "heavy" ones in pocl library,
     * so if the module references images, we do global-only flatten,
     * otherwise the normal flatten.
     *
     * TODO this still requires some solution. With flatten-globals-only,
     * AMDSDK's BlackScholes and BinomialOption regress heavily. OTOH the
     * performance of Mandelbrot is about 30% higher with flatten-globals
     * compared to flatten-inline-all (likely because inline-all compiled
     * Mandelbrot kernel exceeds L1 instruction cache size).
     */
    if (refs_images) {
      POCL_MSG_PRINT_LLVM(
          "Module references images, using flatten-globals-only\n");
      passes.push_back("flatten-globals");
#ifndef LLVM_3_9
      passes.push_back("inline");
#endif
    } else {
      passes.push_back("flatten-inline-all");
    }
  }
  passes.push_back("always-inline");
#ifndef LLVM_OLDER_THAN_4_0
  // It should be now safe to run -O3 over the single work-item kernel
  // as the barrier has the attributes preventing illegal motions and
  // duplication. Let's do it to clean up the code for later passes.
  // Especially the WI context structures get needlessly bloated in case there
  // is dead code lying around.
  passes.push_back("STANDARD_OPTS");
#else
  // Just clean up any unused globals.
  passes.push_back("globaldce");
#endif

  if (!SPMDDevice) {
    passes.push_back("simplifycfg");
    passes.push_back("loop-simplify");
    passes.push_back("uniformity");
    passes.push_back("phistoallocas");
    passes.push_back("isolate-regions");
    passes.push_back("implicit-loop-barriers");
    passes.push_back("implicit-cond-barriers");
    passes.push_back("loop-barriers");
    passes.push_back("barriertails");
    passes.push_back("barriers");
    passes.push_back("isolate-regions");
    passes.push_back("wi-aa");
    passes.push_back("workitemrepl");
    //passes.push_back("print-module");
    passes.push_back("workitemloops");
    // Remove the (pseudo) barriers.   They have no use anymore due to the
    // work-item loop control taking care of them.
    passes.push_back("remove-barriers");
  }
  // Add the work group launcher functions and privatize the pseudo variable
  // (local id) accesses.
  if (device->workgroup_pass)
    passes.push_back("workgroup");

  // Attempt to move all allocas to the entry block to avoid the need for
  // dynamic stack which is problematic for some architectures.
  passes.push_back("allocastoentry");

#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  // Convert the semantical OpenCL address space IDs to the ones of the target.
  passes.push_back("target-address-spaces");
#endif

  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running
  // the standard LLVM optimizations.
  passes.push_back("simplifycfg");

#if 0
  passes.push_back("print-module");
  passes.push_back("dot-cfg");
#endif

  if (currentWgMethod == "loopvec")
    passes.push_back("scalarizer");

  passes.push_back("instcombine");
  passes.push_back("STANDARD_OPTS");
  passes.push_back("instcombine");

  // Now actually add the listed passes to the PassManager.
  for(unsigned i = 0; i < passes.size(); ++i) {
      // This is (more or less) -O3.
      if (passes[i] == "STANDARD_OPTS")
        {
          PassManagerBuilder Builder;
          Builder.OptLevel = 3;
          Builder.SizeLevel = 0;

          // These need to be setup in addition to invoking the passes
          // to get the vectorizers initialized properly.
          if (currentWgMethod == "loopvec") {
            Builder.LoopVectorize = true;
            Builder.SLPVectorize = true;
          }
          Builder.populateModulePassManager(*Passes);
          continue;
        }

        const PassInfo *PIs = Registry->getPassInfo(StringRef(passes[i]));
        if (PIs) {
          //std::cout << "-"<<passes[i] << " ";
          Pass *thispass = PIs->createPass();
          Passes->add(thispass);
        }
      else
        {
          std::cerr << "Failed to create kernel compiler pass " << passes[i] << std::endl;
          POCL_ABORT("FAIL");
        }
    }

    kernel_pass_map[device] = Passes;
    return *Passes;
}

// Defined in llvmopencl/WorkitemHandler.cc
namespace pocl {
    extern size_t WGLocalSizeX;
    extern size_t WGLocalSizeY;
    extern size_t WGLocalSizeZ;
    extern bool WGDynamicLocalSize;
} 

/**
 * Return the OpenCL C built-in function library bitcode
 * for the given device.
 */
static llvm::Module*
kernel_library
(cl_device_id device)
{
  Triple triple(device->llvm_target_triplet);

  if (kernelLibraryMap.find(device) != kernelLibraryMap.end())
    return kernelLibraryMap[device];

  const char *subdir = "host";
  bool is_host = true;
#ifdef TCE_AVAILABLE
  if (triple.getArch() == Triple::tce) {
    subdir = "tce";
    is_host = false;
  }
#endif
#ifdef BUILD_HSA
  if (triple.getArch() == Triple::hsail64) {
    subdir = "hsail64";
    is_host = false;
  }
#endif
#ifdef AMDGCN_ENABLED
  if (triple.getArch == Triple::amdgcn) {
    subdir = "amdgcn";
    is_host = false;
  }
#endif
#ifdef BUILD_CUDA
  if (triple.getArch() == Triple::nvptx ||
      triple.getArch() == Triple::nvptx64) {
    subdir = "cuda";
    is_host = false;
  }
#endif

  // TODO sync with Nat Ferrus' indexed linking
  std::string kernellib;
  std::string kernellib_fallback;
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    kernellib = BUILDDIR;
    kernellib += "/lib/kernel/";
    kernellib += subdir;
    // TODO: get this from the TCE target triplet
    kernellib += "/kernel-";
    kernellib += device->llvm_target_triplet;
    if (is_host) {
      kernellib += '-';
      kernellib_fallback = kernellib;
      kernellib_fallback += OCL_KERNEL_TARGET_CPU;
      kernellib_fallback += ".bc";
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
      if (triple.getArch() == Triple::x86_64 ||
          triple.getArch() == Triple::x86)
        kernellib += getX86KernelLibName();
      else
#endif
        kernellib += device->llvm_cpu;
    }
  } else { // POCL_BUILDING == 0, use install dir
    kernellib = PKGDATADIR;
    kernellib += "/kernel-";
    kernellib += device->llvm_target_triplet;
    if (is_host) {
      kernellib += '-';
      kernellib_fallback = kernellib;
      kernellib_fallback += OCL_KERNEL_TARGET_CPU;
      kernellib_fallback += ".bc";
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
      if (triple.getArch() == Triple::x86_64 ||
          triple.getArch() == Triple::x86)
        kernellib += getX86KernelLibName();
      else
#endif
        kernellib += device->llvm_cpu;
    }
  }
  kernellib += ".bc";

  llvm::Module *lib;
  SMDiagnostic Err;

  if (pocl_exists(kernellib.c_str()))
    {
      POCL_MSG_PRINT_LLVM("Using %s as the built-in lib.\n", kernellib.c_str());
      lib = parseIRFile(kernellib.c_str(), Err, *GlobalContext()).release();
    }
  else
    {
      if (is_host && pocl_exists(kernellib_fallback.c_str()))
        {
          POCL_MSG_WARN("Using fallback %s as the built-in lib.\n",
                        kernellib_fallback.c_str());
          lib = parseIRFile(kernellib_fallback.c_str(), Err, *GlobalContext()).release();
        }
      else
        POCL_ABORT("Kernel library file %s doesn't exist.", kernellib.c_str());
    }
  assert (lib != NULL);
  kernelLibraryMap[device] = lib;

  return lib;
}

/* This is used to control the kernel we want to process in the kernel compilation. */
extern cl::opt<std::string> KernelName;

int pocl_llvm_generate_workgroup_function_nowrite(
    cl_device_id device, cl_kernel kernel, size_t local_x, size_t local_y,
    size_t local_z, void **output) {

  int device_i = pocl_cl_device_to_index(kernel->program, device);
  assert(device_i >= 0);

  pocl::WGDynamicLocalSize = (local_x == 0 && local_y == 0 && local_z == 0);

  currentPoclDevice = device;

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

#ifdef DEBUG_POCL_LLVM_API
  printf("### calling the kernel compiler for kernel %s local_x %zu "
         "local_y %zu local_z %zu parallel_filename: %s\n",
         kernel->name, local_x, local_y, local_z, parallel_bc_path);
#endif

  Triple triple(device->llvm_target_triplet);

  SMDiagnostic Err;

  // Link the kernel and runtime library
  llvm::Module *input = NULL;
  if (kernel->program->llvm_irs != NULL &&
      kernel->program->llvm_irs[device_i] != NULL)
    {
#ifdef DEBUG_POCL_LLVM_API
      printf("### cloning the preloaded LLVM IR\n");
#endif
      llvm::Module* p = (llvm::Module*)kernel->program->llvm_irs[device_i];
#ifdef LLVM_OLDER_THAN_3_8
      input = llvm::CloneModule(p);
#else
      input = (llvm::CloneModule(p)).release();
#endif
    }
  else
    {
#ifdef DEBUG_POCL_LLVM_API
      printf("### loading the kernel bitcode from disk\n");
#endif
      char program_bc_path[POCL_FILENAME_LENGTH];
      pocl_cache_program_bc_path(program_bc_path, kernel->program, device_i);
      input = parseIRFile(program_bc_path, Err, *GlobalContext()).release();
    }

  /* Note this is a hack to get SPIR working. We'll be linking the
   * host kernel library (plain LLVM IR) to the SPIR program.bc,
   * so LLVM complains about incompatible DataLayouts. The proper solution
   * would be to generate a SPIR kernel library
   */
  if (triple.getArch() == Triple::x86 || triple.getArch() == Triple::x86_64) {
      if (input->getTargetTriple().substr(0, 6) == std::string("spir64")) {
          input->setTargetTriple(triple.getTriple());
          input->setDataLayout("e-m:e-i64:64-f80:128-n8:16:32:64-S128");
      } else if (input->getTargetTriple().substr(0, 4) == std::string("spir")) {
          input->setTargetTriple(triple.getTriple());
          input->setDataLayout("e-m:e-p:32:32-i64:64-f80:32-n8:16:32-S32");
      }
  }

  /* Now finally run the set of passes assembled above */
  // TODO pass these as parameters instead, this is not thread safe!
  pocl::WGLocalSizeX = local_x;
  pocl::WGLocalSizeY = local_y;
  pocl::WGLocalSizeZ = local_z;
  KernelName = kernel->name;

#ifdef LLVM_OLDER_THAN_3_7
  kernel_compiler_passes(device, input,
                         input->getDataLayout()->getStringRepresentation())
      .run(*input);
#else
  kernel_compiler_passes(device, input,
                         input->getDataLayout().getStringRepresentation())
      .run(*input);
#endif

  assert(output != NULL);
  *output = (void *)input;
  ++numberOfIRs;
  return 0;
}

int pocl_llvm_generate_workgroup_function(cl_device_id device, cl_kernel kernel,
                                          size_t local_x, size_t local_y,
                                          size_t local_z) {

  void *modp = NULL;

  int device_i = pocl_cl_device_to_index(kernel->program, device);
  assert(device_i >= 0);

  char parallel_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_work_group_function_path(parallel_bc_path, kernel->program,
                                      device_i, kernel, local_x, local_y,
                                      local_z);

  if (pocl_exists(parallel_bc_path))
    return CL_SUCCESS;

  char final_binary_path[POCL_FILENAME_LENGTH];
  pocl_cache_final_binary_path(final_binary_path, kernel->program, device_i,
                               kernel, local_x, local_y, local_z);

  if (pocl_exists(final_binary_path))
    return CL_SUCCESS;

  int error = pocl_llvm_generate_workgroup_function_nowrite(
      device, kernel, local_x, local_y, local_z, &modp);
  if (error)
    return error;

  error = pocl_cache_write_kernel_parallel_bc(
      modp, kernel->program, device_i, kernel, local_x, local_y, local_z);

  if (error)
    return error;

  pocl_destroy_llvm_module(modp);
  return error;
}


int
pocl_update_program_llvm_irs(cl_program program,
                             unsigned device_i,
                             cl_device_id device)
{
  SMDiagnostic Err;
  char program_bc_path[POCL_FILENAME_LENGTH];
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  pocl_cache_program_bc_path(program_bc_path, program, device_i);

  if (!pocl_exists(program_bc_path))
    return -1;

  assert(program->llvm_irs[device_i] == nullptr);
  program->llvm_irs[device_i] =
              parseIRFile(program_bc_path, Err, *GlobalContext()).release();
  ++numberOfIRs;
  return 0;
}

void pocl_free_llvm_irs(cl_program program, int device_i)
{
    if (program->llvm_irs[device_i]) {
        llvm::Module *mod = (llvm::Module *)program->llvm_irs[device_i];
        delete mod;
        --numberOfIRs;
        program->llvm_irs[device_i] = NULL;
    }
}

void pocl_llvm_update_binaries (cl_program program) {

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();
  char program_bc_path[POCL_FILENAME_LENGTH];
  void* cache_lock = NULL;
  int error;

  // Dump the LLVM IR Modules to memory buffers. 
  assert (program->llvm_irs != NULL);
#ifdef DEBUG_POCL_LLVM_API        
  printf("### refreshing the binaries of the program %p\n", program);
#endif

   for (size_t i = 0; i < program->num_devices; ++i)
    {
      assert (program->llvm_irs[i] != NULL);
      if (program->binaries[i])
          continue;

      cache_lock = pocl_cache_acquire_writer_lock_i(program, i);
      assert(cache_lock);

      pocl_cache_program_bc_path(program_bc_path, program, i);
      error = pocl_write_module((llvm::Module*)program->llvm_irs[i],
                                 program_bc_path, 1);
      assert(error == 0);

      std::string content;
      llvm::raw_string_ostream sos(content);
      WriteBitcodeToFile((llvm::Module*)program->llvm_irs[i], sos);
      sos.str(); // flush

      size_t n = content.size();
      if (n < program->binary_sizes[i])
        POCL_ABORT("binary size doesn't match the expected value");
      if (program->binaries[i])
          POCL_MEM_FREE(program->binaries[i]);
      program->binaries[i] = (unsigned char *) malloc(n);
      std::memcpy(program->binaries[i], content.c_str(), n);

      pocl_cache_release_lock(cache_lock);
      cache_lock = NULL;

#ifdef DEBUG_POCL_LLVM_API        
      printf("### binary for device %zi was of size %zu\n", i, program->binary_sizes[i]);
#endif

    }
}

/* This is the implementation of the public pocl_llvm_get_kernel_count(),
 * and is used internally also by pocl_llvm_get_kernel_names to
 */
static unsigned
pocl_llvm_get_kernel_count(cl_program program, char **knames,
                           unsigned max_num_krn)
{
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  // TODO: is it safe to assume every device (i.e. the index 0 here)
  // has the same set of programs & kernels?
  llvm::Module *mod = (llvm::Module *) program->llvm_irs[0];

  llvm::NamedMDNode *md = mod->getNamedMetadata("opencl.kernels");
  if (md) {

    if (knames) {
      for (unsigned i=0; i<max_num_krn; i++) {
        assert( md->getOperand(i)->getOperand(0) != NULL);
        llvm::ValueAsMetadata *value =
          dyn_cast<llvm::ValueAsMetadata>(md->getOperand(i)->getOperand(0));
        llvm::Function *k = cast<Function>(value->getValue());
        knames[i] = strdup(k->getName().data());
      }
    }
    return md->getNumOperands();
  }
  // LLVM 3.9 does not use opencl.kernels meta, but kernel_arg_* function meta
  else {
    unsigned kernel_count = 0;
    for (llvm::Module::iterator i = mod->begin(), e = mod->end();
           i != e; ++i) {
      if (i->getMetadata("kernel_arg_access_qual")) {
        if (knames && kernel_count < max_num_krn) {
          knames[kernel_count] = strdup(i->getName().str().c_str());
        }
        ++kernel_count;
      }
    }
    return kernel_count;
  }
}

unsigned
pocl_llvm_get_kernel_count(cl_program program)
{
  return pocl_llvm_get_kernel_count(program, NULL, 0);
}

unsigned
pocl_llvm_get_kernel_names(cl_program program, char **knames,
                           unsigned max_num_krn)
{
  unsigned n = pocl_llvm_get_kernel_count(program, knames, max_num_krn);

  return n;
}

void pocl_destroy_llvm_module(void *modp) {
  llvm::Module *mod = (llvm::Module *)modp;
  if (mod) {
    delete mod;
    --numberOfIRs;
  }
}

/* Run LLVM codegen on input file (parallel-optimized).
 * modp = llvm::Module* of parallel.bc
 * Output native object file (<kernel>.so.o). */
int pocl_llvm_codegen(cl_kernel kernel, cl_device_id device, void *modp,
                      char **output, size_t *output_size) {

  llvm::MutexGuard lockHolder(kernelCompilerLock);

  llvm::Triple triple(device->llvm_target_triplet);
  llvm::TargetMachine *target = GetTargetMachine(device);

  llvm::Module *input = (llvm::Module *)modp;
  assert(input);
  *output = NULL;

  PassManager PM;
#ifdef LLVM_OLDER_THAN_3_7
    llvm::TargetLibraryInfo *TLI = new TargetLibraryInfo(triple);
    PM.add(TLI);
#else
    llvm::TargetLibraryInfoWrapperPass *TLIPass = new TargetLibraryInfoWrapperPass(triple);
    PM.add(TLIPass);
#endif
#ifdef LLVM_OLDER_THAN_3_7
    if (target != NULL) {
      target->addAnalysisPasses(PM);
    }
#endif

    // TODO: get DataLayout from the 'device'
    // TODO: better error check
#ifdef LLVM_OLDER_THAN_3_7
    std::string data;
    llvm::raw_string_ostream sos(data);
    llvm::MCContext *mcc;
    if (target && target->addPassesToEmitMC(PM, mcc, sos))
      return 1;
#else
    SmallVector<char, 4096> data;
    llvm::raw_svector_ostream sos(data);
    if (target && target->addPassesToEmitFile(
        PM, sos, TargetMachine::CGFT_ObjectFile))
      return 1;
#endif

    PM.run(*input);
    std::string o = sos.str(); // flush
    const char *cstr = o.c_str();
    size_t s = o.size();
    *output = (char *)malloc(s);
    *output_size = s;
    memcpy(*output, cstr, s);

    return 0;
}
/* vim: set ts=4 expandtab: */
