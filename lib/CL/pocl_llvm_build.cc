/* pocl_llvm_build.cc: part of pocl's LLVM API which deals with
   producing program.bc

   Copyright (c) 2013 Kalle Raiskila
                 2013-2019 Pekka Jääskeläinen

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

#include <clang/Basic/Diagnostic.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>

#ifndef LLVM_OLDER_THAN_4_0
#include "clang/Lex/PreprocessorOptions.h"
#endif

#include "llvm/LinkAllPasses.h"
#include "llvm/Linker/Linker.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/MutexGuard.h"

#include <iostream>
#include <sstream>

// For some reason including pocl.h before including CodeGenAction.h
// causes an error. Some kind of macro definition issue. To investigate.
#include "pocl.h"
// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros,
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"
#include "linker.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "LLVMUtils.h"

using namespace clang;
using namespace llvm;

POP_COMPILER_DIAGS

/* Global pocl device to be used by passes if needed */
cl_device_id currentPoclDevice = NULL;


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
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) != 0)
    return 0;

  FrontendInputFile const& file = fe.Inputs.front();
  if (file.isFile() && !file.isSystem()) {
    return pocl_remove(file.getFile().str().c_str());
  } else {
    return 0; // nothing to do
  }

}

static void appendToProgramBuildLog(cl_program program, unsigned device_i,
                                    std::string &s) {
  if (!s.empty()) {
    POCL_MSG_ERR("%s", s.c_str());
    /* this may not actually write anything if the buildhash is invalid,
     * but program->build_log still gets written.  */
    pocl_cache_append_to_buildlog(program, device_i, s.c_str(), s.size());
    if (program->build_log[device_i]) {
      size_t len = strlen(program->build_log[device_i]);
      size_t len2 = strlen(s.c_str());
      char *newlog = (char *)malloc(len + len2 + 1);
      memcpy(newlog, program->build_log[device_i], len);
      memcpy(newlog + len, s.c_str(), len2);
      newlog[len + len2] = 0;
      POCL_MEM_FREE(program->build_log[device_i]);
      program->build_log[device_i] = newlog;
    } else
      program->build_log[device_i] = strdup(s.c_str());
  }
}

static void get_build_log(cl_program program,
                         unsigned device_i,
                         std::stringstream &ss_build_log,
                         clang::TextDiagnosticBuffer *diagsBuffer,
                         const SourceManager *SM)
{
  for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->err_begin(),
         e = diagsBuffer->err_end(); i != e; ++i)
  {
    ss_build_log << "error: "
                 << (SM == nullptr ? "" : (i->first.printToString(*SM) + ": "))
                 << i->second << std::endl;
  }
  for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->warn_begin(),
         e = diagsBuffer->warn_end(); i != e; ++i)
  {
    ss_build_log << "warning: "
                 << (SM == nullptr ? "" : (i->first.printToString(*SM) + ": "))
                 << i->second << std::endl;
  }

  std::string log = ss_build_log.str();
  appendToProgramBuildLog(program, device_i, log);
}

static llvm::Module *getKernelLibrary(cl_device_id device);

int pocl_llvm_build_program(cl_program program,
                            unsigned device_i,
                            const char *user_options_cstr,
                            char *program_bc_path,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names,
                            int linking_program)

{
  char tempfile[POCL_FILENAME_LENGTH];
  tempfile[0] = 0;
  llvm::Module **mod = NULL;
  char temp_include_dir[POCL_FILENAME_LENGTH];
  std::string user_options(user_options_cstr ? user_options_cstr : "");
  size_t n = 0;
  int error;

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  if (num_input_headers > 0) {
    error = pocl_cache_create_tempdir(temp_include_dir);
    if(error)
      {
        POCL_MSG_ERR ("pocl_cache_create_tempdir (%s)"
                      " failed with %i\n", temp_include_dir, error);
        return error;
      }
    std::string tempdir(temp_include_dir);

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

  ss << "-DPOCL_DEVICE_ADDRESS_BITS=" << device->address_bits << " ";
#ifndef LLVM_OLDER_THAN_4_0
  ss << "-D__USE_CLANG_OPENCL_C_H ";
#endif

  ss << "-xcl ";
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
  while (i != end) {
    itemstrs.push_back(*i);
    ++i;
  }

  for (unsigned idx = 0; idx < itemstrs.size(); idx++) {
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
    get_build_log(program, device_i, ss_build_log, diagsBuffer,
                  CI.hasSourceManager() ? &CI.getSourceManager() : nullptr);
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

  la->setStackProtector(LangOptions::StackProtectorMode::SSPOff);

  la->PICLevel = PICLevel::BigPIC;
  la->PIE = 1;

  std::string IncludeRoot;
  std::string KernelH;
  std::string BuiltinRenamesH;
  std::string PoclTypesH;

#ifdef ENABLE_POCL_BUILDING
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    IncludeRoot = SRCDIR;
#else
  if (0) {
#endif
  } else {
    IncludeRoot = POCL_INSTALL_PRIVATE_DATADIR;
  }
  KernelH = IncludeRoot + "/include/_kernel.h";
  BuiltinRenamesH = IncludeRoot + "/include/_builtin_renames.h";
  PoclTypesH = IncludeRoot + "/include/pocl_types.h";

  po.Includes.push_back(PoclTypesH);
  po.Includes.push_back(BuiltinRenamesH);
#ifndef LLVM_OLDER_THAN_4_0
  // Use Clang's opencl-c.h header.
  {
#if (!defined(LLVM_OLDER_THAN_8_0)) && (!defined(LLVM_8_0))
      std::string ClangResourcesDir = driver::Driver::GetResourcesPath(CLANG);
#else
      DiagnosticsEngine Diags{new DiagnosticIDs, new DiagnosticOptions};
      driver::Driver TheDriver(CLANG, "", Diags);
      std::string ClangResourcesDir = TheDriver.ResourceDir;
#endif
      po.Includes.push_back(ClangResourcesDir + "/include/opencl-c.h");
  }
#endif
  po.Includes.push_back(KernelH);
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
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  cg.VerifyModule = false;
#else
  cg.VerifyModule = true;
#endif

  PreprocessorOutputOptions &poo = pocl_build.getPreprocessorOutputOpts();
  poo.ShowCPP = 1;
  poo.ShowComments = 0;
  poo.ShowLineMarkers = 0;
  poo.ShowMacroComments = 0;
  poo.ShowMacros = 1;
  poo.RewriteIncludes = 0;

  pocl_cache_tempname(tempfile, ".preproc.cl", NULL);
  fe.OutputFile.assign(tempfile);

  bool success = true;
  clang::PrintPreprocessedAction Preprocess;
  success = CI.ExecuteAction(Preprocess);
  char *PreprocessedOut = nullptr;
  uint64_t PreprocessedSize = 0;

  if (success) {
    pocl_read_file(tempfile, &PreprocessedOut, &PreprocessedSize);
  }
  /* always remove preprocessed output - the sources are in different files */
  pocl_remove(tempfile);

  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0) {
    if (num_input_headers > 0)
      pocl_rm_rf(temp_include_dir);
  }

  if (PreprocessedOut == nullptr) {
    pocl_cache_create_program_cachedir(program, device_i, NULL, 0,
                                       program_bc_path);
    get_build_log(program, device_i, ss_build_log, diagsBuffer,
                  CI.hasSourceManager() ? &CI.getSourceManager() : nullptr);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  pocl_cache_create_program_cachedir(program, device_i, PreprocessedOut,
                                     static_cast<size_t>(PreprocessedSize), program_bc_path);

  POCL_MEM_FREE(PreprocessedOut);

  unlink_source(fe);

  if (pocl_exists(program_bc_path))
    return CL_SUCCESS;

  // TODO: use pch: it is possible to disable the strict checking for
  // the compilation flags used to compile it and the current translation
  // unit via the preprocessor options directly.
  llvm::LLVMContext &c = GlobalContext();
  clang::EmitLLVMOnlyAction EmitLLVM(&c);
  success = CI.ExecuteAction(EmitLLVM);

  get_build_log(program, device_i, ss_build_log, diagsBuffer, &CI.getSourceManager());

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

  if ((*mod)->getModuleFlag("PIC Level") == nullptr)
    (*mod)->setPICLevel(PICLevel::BigPIC);
  if ((*mod)->getModuleFlag("PIE Level") == nullptr)
    (*mod)->setPIELevel(PIELevel::Large);

  // link w kernel lib, but not if we're called from clCompileProgram()
  // Later this should be replaced with indexed linking of source code
  // and/or bitcode for each kernel.
  if (linking_program) {
    currentPoclDevice = device;
    llvm::Module *libmodule = getKernelLibrary(device);
    assert(libmodule != NULL);
    std::string log("Error(s) while linking: \n");
    if (link(*mod, libmodule, log)) {
      appendToProgramBuildLog(program, device_i, log);
      std::string msg = getDiagString();
      appendToProgramBuildLog(program, device_i, msg);
      delete *mod;
      *mod = nullptr;
      --numberOfIRs;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc. Its required in clBuildProgram */
  error = pocl_write_module(*mod, program_bc_path, 0);
  if(error)
    return error;

  /* To avoid writing & reading the same back,
   * save program->binaries[i]
   */
  std::string content;
  writeModuleIR(*mod, content);

  if (program->binaries[device_i])
    POCL_MEM_FREE(program->binaries[device_i]);

  n = content.size();
  program->binary_sizes[device_i] = n;
  program->binaries[device_i] = (unsigned char *) malloc(n);
  std::memcpy(program->binaries[device_i], content.c_str(), n);

  return CL_SUCCESS;
}

int pocl_llvm_link_program(cl_program program, unsigned device_i,
                           char *program_bc_path, cl_uint num_input_programs,
                           unsigned char **cur_device_binaries,
                           size_t *cur_device_binary_sizes, void **cur_llvm_irs,
                           int create_library, int spir) {

  std::string concated_binaries;
  llvm::Module *linked_module = nullptr;
  size_t n = 0, i;
  cl_device_id device = program->devices[device_i];
  llvm::Module **modptr = (llvm::Module **)&program->llvm_irs[device_i];
  int error;

  currentPoclDevice = device;
  llvm::Module *libmodule = getKernelLibrary(device);
  assert(libmodule != NULL);

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  if (spir) {
#ifdef ENABLE_SPIR

    POCL_RETURN_ERROR_ON ((device->endian_little == CL_FALSE),
                         CL_LINK_PROGRAM_FAILURE,
                         "SPIR is only supported on little-endian devices\n");

    linked_module = parseModuleIRMem((char *)program->binaries[device_i],
                                     program->binary_sizes[device_i]);

    const std::string &spir_triple = linked_module->getTargetTriple();
    size_t spir_addrbits = Triple(spir_triple).isArch64Bit() ? 64 : 32;

    if (device->address_bits != spir_addrbits) {
        delete linked_module;
        POCL_RETURN_ERROR_ON (1, CL_LINK_PROGRAM_FAILURE,
                       "Device address bits != SPIR binary triple address "
                       "bits, device: %s / module: %s\n",
                       device->llvm_target_triplet, spir_triple.c_str());
    }

    /* Note this is a hack to get SPIR working. We'll be linking the
     * host kernel library (plain LLVM IR) to the SPIR program.bc,
     * so LLVM complains about incompatible DataLayouts.
     */
    linked_module->setTargetTriple(libmodule->getTargetTriple());
    linked_module->setDataLayout(libmodule->getDataLayout());
    if (linked_module->getModuleFlag("PIC Level") == nullptr)
      linked_module->setPICLevel(PICLevel::BigPIC);
    if (linked_module->getModuleFlag("PIE Level") == nullptr)
      linked_module->setPIELevel(PIELevel::Large);

#else
    POCL_MSG_ERR("SPIR not supported\n");
    return CL_LINK_PROGRAM_FAILURE;
#endif
  } else {

#ifdef LLVM_OLDER_THAN_3_8
  llvm::Module *mod =
      new llvm::Module(StringRef("linked_program"), GlobalContext());
#else
  std::unique_ptr<llvm::Module> mod(
      new llvm::Module(StringRef("linked_program"), GlobalContext()));
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
#elif LLVM_OLDER_THAN_7_0
    if (Linker::linkModules(*mod, llvm::CloneModule(p))) {
#else
    if (Linker::linkModules(*mod, llvm::CloneModule(*p))) {
#endif
      std::string msg = getDiagString();
      appendToProgramBuildLog(program, device_i, msg);
      return CL_LINK_PROGRAM_FAILURE;
    }
  }

#ifdef LLVM_OLDER_THAN_3_8
  linked_module = mod;
#else
  linked_module = mod.release();
#endif
  }

  if (linked_module == nullptr)
    return CL_LINK_PROGRAM_FAILURE;

  if (*modptr != nullptr) {
    delete *modptr;
    --numberOfIRs;
    *modptr = nullptr;
  }

  if (!create_library) {
    // linked all the programs together, now link in the kernel library
    std::string log("Error(s) while linking: \n");
    if (link(linked_module, libmodule, log)) {
      appendToProgramBuildLog(program, device_i, log);
      std::string msg = getDiagString();
      appendToProgramBuildLog(program, device_i, msg);
      delete linked_module;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  *modptr = linked_module;
  ++numberOfIRs;

  /* TODO currently cached on concated binary contents (in undefined order),
     this is not terribly useful (but we have to store it somewhere..) */
  error = pocl_cache_create_program_cachedir(program, device_i,
                                     concated_binaries.c_str(),
                                     concated_binaries.size(),
                                     program_bc_path);
  if (error)
    {
      POCL_MSG_ERR ("pocl_cache_create_program_cachedir(%s)"
                    " failed with %i\n", program_bc_path, error);
      return error;
    }

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc. Its required in clBuildProgram */
  error = pocl_write_module(linked_module, program_bc_path, 0);
  if (error)
    return error;

  /* To avoid writing & reading the same back,
   * save program->binaries[i]
   */
  std::string content;
  writeModuleIR(linked_module, content);

  if (program->binaries[device_i])
    POCL_MEM_FREE(program->binaries[device_i]);

  n = content.size();
  program->binary_sizes[device_i] = n;
  program->binaries[device_i] = (unsigned char *)malloc(n);
  std::memcpy(program->binaries[device_i], content.c_str(), n);

  return CL_SUCCESS;
}

/* for "distro" style kernel libs, return which kernellib to use, at runtime */
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
const char *getX86KernelLibName() {
  StringMap<bool> Features;
  llvm::sys::getHostCPUFeatures(Features);
  const char *res = NULL;

  if (Features["sse2"])
    res = "sse2";
  else
    POCL_ABORT("Pocl on x86_64 requires at least SSE2\n");
  if (Features["ssse3"] && Features["cx16"])
    res = "ssse3";
  if (Features["sse4.1"] && Features["cx16"])
    res = "sse41";
  if (Features["avx"] && Features["cx16"] && Features["popcnt"])
    res = "avx";
  if (Features["avx"] && Features["cx16"] && Features["popcnt"] && Features["f16c"])
    res = "avx_f16c";
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


static std::map<cl_device_id, llvm::Module *> kernelLibraryMap;

/**
 * Return the OpenCL C built-in function library bitcode
 * for the given device.
 */
static llvm::Module* getKernelLibrary(cl_device_id device)
{
  Triple triple(device->llvm_target_triplet);

  if (kernelLibraryMap.find(device) != kernelLibraryMap.end())
    return kernelLibraryMap[device];

  const char *subdir = "host";
  bool is_host = true;
  // TODO: move this to the device layer, a property to ask for
  // the kernel builtin bitcode library name, including its subdir
#ifdef TCE_AVAILABLE
  if (triple.getArch() == Triple::tce || triple.getArch() == Triple::tcele) {
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

  std::string kernellib;
  std::string kernellib_fallback;
#ifdef ENABLE_POCL_BUILDING
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    kernellib = BUILDDIR;
    kernellib += "/lib/kernel/";
    kernellib += subdir;
  } else // POCL_BUILDING == 0, use install dir
#endif
  kernellib = POCL_INSTALL_PRIVATE_DATADIR;
  kernellib += "/kernel-";
  kernellib += device->llvm_target_triplet;
  if (is_host) {
    kernellib += '-';
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
    kernellib += getX86KernelLibName();
#else
    kernellib_fallback = kernellib;
    kernellib_fallback += OCL_KERNEL_TARGET_CPU;
    kernellib_fallback += ".bc";
    kernellib += device->llvm_cpu;
#endif
  }
  kernellib += ".bc";

  llvm::Module *lib;

  if (pocl_exists(kernellib.c_str()))
    {
      POCL_MSG_PRINT_LLVM("Using %s as the built-in lib.\n", kernellib.c_str());
      lib = parseModuleIR(kernellib.c_str());
    }
  else
    {
#ifndef KERNELLIB_HOST_DISTRO_VARIANTS
      if (is_host && pocl_exists(kernellib_fallback.c_str()))
        {
          POCL_MSG_WARN("Using fallback %s as the built-in lib.\n",
                        kernellib_fallback.c_str());
          lib = parseModuleIR(kernellib_fallback.c_str());
        }
      else
#endif
        POCL_ABORT("Kernel library file %s doesn't exist.\n", kernellib.c_str());
    }
  assert (lib != NULL);
  kernelLibraryMap[device] = lib;

  return lib;
}

void cleanKernelLibrary() {
  for (auto i = kernelLibraryMap.begin(), e = kernelLibraryMap.end();
       i != e; ++i) {
    delete (llvm::Module *)i->second;
  }
  kernelLibraryMap.clear();
}

#ifndef LLVM_OLDER_THAN_5_0
/**
 * Invoke the Clang compiler through its Driver API.
 *
 * @param Device the device of which toolchain to use.
 * @param Args the command line arguments that would be passed to Clang
 *             (a NULL terminated list). Args[0] should be the path to
 *             the Clang binary.
 * @return 0 on success, error code otherwise.
 */
int pocl_invoke_clang(cl_device_id Device, const char** Args) {

  // Borrowed from driver.cpp (clang driver). We do not really care about
  // diagnostics, but just want to get the compilation command invoked with
  // the target's toolchain as defined in Clang.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions;

  TextDiagnosticPrinter *DiagClient =
    new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  driver::Driver TheDriver(CLANG, Device->llvm_target_triplet, Diags);

  const char **ArgsEnd = Args;
  while (*ArgsEnd++ != nullptr) {}

  llvm::ArrayRef<const char*> ArgsArray(Args, ArgsEnd);

  std::unique_ptr<driver::Compilation> C(
    TheDriver.BuildCompilation(ArgsArray));

  if (C && !C->containsError()) {
    SmallVector<std::pair<int, const driver::Command *>, 4> FailingCommands;
    return TheDriver.ExecuteCompilation(*C, FailingCommands);
  } else {
    return -1;
  }

}
#endif
