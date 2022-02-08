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
#include "clang/Lex/PreprocessorOptions.h"

#ifdef LLVM_OLDER_THAN_10_0
#include "llvm/ADT/ArrayRef.h"
#endif

#include "llvm/LinkAllPasses.h"
#include "llvm/Linker/Linker.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#ifndef LLVM_OLDER_THAN_11_0
#include "llvm/Support/Host.h"
#endif

#ifdef ENABLE_RELOCATION

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#include <dlfcn.h>

#endif

#include <iostream>
#include <sstream>
#include <regex>

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
#include "pocl_util.h"

using namespace clang;
using namespace llvm;

POP_COMPILER_DIAGS



//#define DEBUG_POCL_LLVM_API

#if defined(DEBUG_POCL_LLVM_API) && defined(NDEBUG)
#undef NDEBUG
#include <cassert>
#endif


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

static llvm::Module *getKernelLibrary(cl_device_id device,
                                      PoclLLVMContextData *llvm_ctx);

static std::string getPoclPrivateDataDir() {
#ifdef ENABLE_RELOCATION
    Dl_info info;
    if (dladdr((void*)getPoclPrivateDataDir, &info)) {
        char const * soname = info.dli_fname;
        std::string result = std::string(soname);
        size_t last_slash = result.rfind('/');
        result = result.substr(0, last_slash+1);
        if (result.size() > 0) {
            result += POCL_INSTALL_PRIVATE_DATADIR_REL;
            return result;
        }
    }
#endif
    return POCL_INSTALL_PRIVATE_DATADIR;
}

int pocl_llvm_build_program(cl_program program,
                            unsigned device_i,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names,
                            int linking_program)

{
  char tempfile[POCL_FILENAME_LENGTH];
  char program_bc_path[POCL_FILENAME_LENGTH];
  tempfile[0] = 0;
  llvm::Module *mod = nullptr;
  char temp_include_dir[POCL_FILENAME_LENGTH];
  std::string user_options(program->compiler_options ? program->compiler_options
                                                     : "");
  size_t n = 0;
  int error;
  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

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

#if !(defined(__x86_64__) && defined(__GNUC__))
  if (program->flush_denorms) {
    POCL_MSG_WARN("flush to zero is currently only implemented for "
                  "x86-64 & gcc/clang, ignoring flag\n");
  }
#endif

  std::string cl_ext;
  if (extensions.size() > 0) {
    size_t e_start = 0, e_end = 0;
    while (e_end < std::string::npos) {
      e_end = extensions.find(' ', e_start);
      llvm::StringRef tok = extensions.slice(e_start, e_end);
      e_start = e_end + 1;
      ss << "-D" << tok.str() << " ";
      cl_ext += "+";
      cl_ext += tok.str();
      cl_ext += ",";
    }
  }
  if (!cl_ext.empty()) {
    cl_ext.back() = ' '; // replace last "," with space
    ss << "-cl-ext=-all," << cl_ext;
  }

  /* temp dir takes preference */
  if (num_input_headers > 0)
    ss << "-I" << temp_include_dir << " ";

  if (device->has_64bit_long)
    ss << "-Dcl_khr_int64 ";

  ss << "-DPOCL_DEVICE_ADDRESS_BITS=" << device->address_bits << " ";
  ss << "-D__USE_CLANG_OPENCL_C_H ";
#ifndef LLVM_OLDER_THAN_13_0
  ss << "-Dreserve_id_t=unsigned ";
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

  char WSReplacementChar = 0;

  char *TempOptions = (char *) malloc (ss.str().length() + 1);

  memset (TempOptions, 0, ss.str().length() + 1);
  strncpy (TempOptions, ss.str().c_str(), ss.str().length());

  if (pocl_escape_quoted_whitespace (TempOptions, &WSReplacementChar) == -1)
  {
    POCL_MEM_FREE (TempOptions);
    return CL_INVALID_BUILD_OPTIONS;
  }

  std::istringstream iss(TempOptions);
  std::vector<const char *> itemcstrs;
  std::vector<std::string> itemstrs;

  std::string s;

  while (iss >> s)
  {
    // if needed, put back whitespace
    if (WSReplacementChar != 0)
    {
      if (s.find(WSReplacementChar) != std::string::npos)
      {
        std::replace(s.begin(), s.end(), WSReplacementChar, ' ');
      }
    }

    // if quoted, remove it to make compiler happy
    if (s.find("\"") != std::string::npos)
    {
      std::regex Target("\"");
      std::string Replacement = " ";
      s = std::regex_replace(s, Target, Replacement);
    }

    itemstrs.push_back(s);
  }

  for (unsigned idx = 0; idx < itemstrs.size(); idx++) {
    // note: if itemstrs is modified after this, itemcstrs will be full
    // of invalid pointers! Could make copies, but would have to clean up then...
    itemcstrs.push_back(itemstrs[idx].c_str());
  }

  POCL_MEM_FREE (TempOptions);

#ifdef DEBUG_POCL_LLVM_API
  // TODO: for some reason the user_options are replicated,
  // they appear twice in a row in the output
  std::cerr << "### options: " << ss.str()
            << "user_options: " << user_options << std::endl;
#endif

  if (!CompilerInvocation::CreateFromArgs(
          pocl_build,
#ifndef LLVM_OLDER_THAN_10_0
          ArrayRef<const char *>(itemcstrs.data(),
                                 itemcstrs.data() + itemcstrs.size()),
#else
          itemcstrs.data(), itemcstrs.data() + itemcstrs.size(),
#endif
          diags)) {
    pocl_cache_create_program_cachedir(program, device_i, program->source,
                                       strlen(program->source),
                                       program_bc_path);
    get_build_log(program, device_i, ss_build_log, diagsBuffer,
                  CI.hasSourceManager() ? &CI.getSourceManager() : nullptr);
    return CL_INVALID_BUILD_OPTIONS;
  }

  LangOptions *la = pocl_build.getLangOpts();
  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();

  pocl_build.setLangDefaults(*la,
#ifndef LLVM_OLDER_THAN_10_0
                             clang::InputKind(clang::Language::OpenCL),
#else
                             clang::InputKind::OpenCL,
#endif
                             triple,
#ifndef LLVM_OLDER_THAN_12_0
                             po.Includes,
#else
                             po,
#endif
                             clang::LangStandard::lang_opencl12);

  // LLVM 3.3 and older do not set that char is signed which is
  // defined by the OpenCL C specs (but not by C specs).
  la->CharIsSigned = true;

  // the per-file types don't seem to override this
  la->OpenCLVersion = cl_std_i;
  la->FakeAddressSpaceMap = false;
  la->Blocks = true; //-fblocks
  la->MathErrno = false; // -fno-math-errno
  la->NoBuiltin = true;  // -fno-builtin
  la->AsmBlocks = true;  // -fasm (?)

  la->setStackProtector(LangOptions::StackProtectorMode::SSPOff);

  la->PICLevel = PICLevel::BigPIC;
#ifdef __PPC64__
  la->PIE = 0;
#else
  la->PIE = 1;
#endif

  std::string IncludeRoot;
  std::string KernelH;
  std::string BuiltinRenamesH;
  std::string PoclTypesH;
  std::string ClangResourceDir;

#ifdef ENABLE_POCL_BUILDING
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    IncludeRoot = SRCDIR;
#else
  if (0) {
#endif
  } else {
    IncludeRoot = getPoclPrivateDataDir();
#ifdef ENABLE_RELOCATION
    ClangResourceDir = IncludeRoot;
#endif
  }
  if (ClangResourceDir.empty()) {     
#ifndef LLVM_OLDER_THAN_9_0
    ClangResourceDir = driver::Driver::GetResourcesPath(CLANG);
#else
    DiagnosticsEngine Diags{new DiagnosticIDs, new DiagnosticOptions};
    driver::Driver TheDriver(CLANG, "", Diags);
    ClangResourceDir = TheDriver.ResourceDir;
#endif
  }
  KernelH = IncludeRoot + "/include/_kernel.h";
  BuiltinRenamesH = IncludeRoot + "/include/_builtin_renames.h";
  PoclTypesH = IncludeRoot + "/include/pocl_types.h";

  po.Includes.push_back(PoclTypesH);
  po.Includes.push_back(BuiltinRenamesH);
  // Use Clang's opencl-c.h header.
#ifndef LLVM_OLDER_THAN_9_0
  po.Includes.push_back(ClangResourceDir + "/include/opencl-c-base.h");
#endif
  po.Includes.push_back(ClangResourceDir + "/include/opencl-c.h");
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

  // Read input source to clang::FrontendOptions.
  // The source is contained in the program->source array,
  // but if debugging option is enabled in the kernel compiler
  // we need to dump the file to disk first for the debugger
  // to find it.
  char source_file[POCL_FILENAME_LENGTH];
  POCL_RETURN_ERROR_ON(pocl_cache_write_program_source(source_file, program),
                       CL_OUT_OF_HOST_MEMORY, "Could not write program source");
  fe.Inputs.push_back(
      FrontendInputFile(source_file,
#ifndef LLVM_OLDER_THAN_10_0
                        clang::InputKind(clang::Language::OpenCL)
#else
                        clang::InputKind::OpenCL
#endif
                            ));

  CodeGenOptions &cg = pocl_build.getCodeGenOpts();
  cg.EmitOpenCLArgMetadata = true;
  cg.StackRealignment = true;
  if (!device->spmd) {
    // Let the vectorizer or another optimization pass unroll the loops,
    // in case it sees beneficial.
    cg.UnrollLoops = false;
    // Lets leave vectorization to later compilation phase
    cg.VectorizeLoop = false;
    cg.VectorizeSLP = false;
  }
  cg.VerifyModule = true;

  PreprocessorOutputOptions &poo = pocl_build.getPreprocessorOutputOpts();
  poo.ShowCPP = 1;
  poo.ShowComments = 0;
  poo.ShowLineMarkers = 0;
  poo.ShowMacroComments = 0;
  poo.ShowMacros = 1;
  poo.RewriteIncludes = 0;

  error = pocl_cache_tempname(tempfile, ".preproc.cl", NULL);
  assert(error == 0);
  fe.OutputFile.assign((const char *)tempfile);

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
    pocl_cache_create_program_cachedir(program, device_i, program->source,
                                       strlen(program->source),
                                       program_bc_path);
    get_build_log(program, device_i, ss_build_log, diagsBuffer,
                  CI.hasSourceManager() ? &CI.getSourceManager() : nullptr);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  pocl_cache_create_program_cachedir(program, device_i, PreprocessedOut,
                                     static_cast<size_t>(PreprocessedSize), program_bc_path);

  POCL_MEM_FREE(PreprocessedOut);

  unlink_source(fe);

  if (pocl_exists(program_bc_path)) {
    char *binary = nullptr;
    uint64_t fsize;
    /* Read binaries from program.bc to memory */
    if (program->binaries[device_i] != nullptr) {
      POCL_MEM_FREE(program->binaries[device_i]);
      program->binary_sizes[device_i] = 0;
    }
    int r = pocl_read_file(program_bc_path, &binary, &fsize);
    POCL_RETURN_ERROR_ON(r, CL_BUILD_ERROR,
                         "Failed to read binaries from program.bc to "
                         "memory: %s\n",
                         program_bc_path);

    program->binary_sizes[device_i] = (size_t)fsize;
    program->binaries[device_i] = (unsigned char *)binary;

    mod = (llvm::Module *)program->data[device_i];
    if (mod != nullptr) {
      delete mod;
      program->data[device_i] = nullptr;
      --llvm_ctx->number_of_IRs;
    }

    program->data[device_i] = parseModuleIR(program_bc_path, llvm_ctx->Context);
    assert(program->data[device_i]);
    ++llvm_ctx->number_of_IRs;

    return CL_SUCCESS;
  }

  // TODO: use pch: it is possible to disable the strict checking for
  // the compilation flags used to compile it and the current translation
  // unit via the preprocessor options directly.
  clang::EmitLLVMOnlyAction EmitLLVM(llvm_ctx->Context);
  success = CI.ExecuteAction(EmitLLVM);

  get_build_log(program, device_i, ss_build_log, diagsBuffer, &CI.getSourceManager());

  if (!success)
    return CL_BUILD_PROGRAM_FAILURE;

  mod = (llvm::Module *)program->data[device_i];
  if (mod != nullptr) {
    delete mod;
    --llvm_ctx->number_of_IRs;
  }

  mod = EmitLLVM.takeModule().release();
  if (mod == nullptr)
    return CL_BUILD_PROGRAM_FAILURE;
  else
    ++llvm_ctx->number_of_IRs;

  if (mod->getModuleFlag("PIC Level") == nullptr)
    mod->setPICLevel(PICLevel::BigPIC);
#ifndef __PPC64__
  if (mod->getModuleFlag("PIE Level") == nullptr)
    mod->setPIELevel(PIELevel::Large);
#endif

  // link w kernel lib, but not if we're called from clCompileProgram()
  // Later this should be replaced with indexed linking of source code
  // and/or bitcode for each kernel.
  if (linking_program) {
    llvm::Module *libmodule = getKernelLibrary(device, llvm_ctx);
    assert(libmodule != NULL);
    std::string log("Error(s) while linking: \n");
    if (link(mod, libmodule, log, device->global_as_id,
             device->device_aux_functions)) {
      appendToProgramBuildLog(program, device_i, log);
      std::string msg = getDiagString(ctx);
      appendToProgramBuildLog(program, device_i, msg);
      delete mod;
      mod = nullptr;
      --llvm_ctx->number_of_IRs;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  program->data[device_i] = mod;

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc */
  error = pocl_write_module(mod, program_bc_path, 0);
  if(error)
    return error;

  /* To avoid writing & reading the same back,
   * save program->binaries[i]
   */
  std::string content;
  writeModuleIRtoString(mod, content);

  if (program->binaries[device_i])
    POCL_MEM_FREE(program->binaries[device_i]);

  n = content.size();
  program->binary_sizes[device_i] = n;
  program->binaries[device_i] = (unsigned char *) malloc(n);
  std::memcpy(program->binaries[device_i], content.c_str(), n);

  return CL_SUCCESS;
}

int pocl_llvm_link_program(cl_program program, unsigned device_i,
                           cl_uint num_input_programs,
                           unsigned char **cur_device_binaries,
                           size_t *cur_device_binary_sizes, void **cur_llvm_irs,
                           int link_program, int spir) {

  char program_bc_path[POCL_FILENAME_LENGTH];
  std::string concated_binaries;
  llvm::Module *linked_module = nullptr;
  size_t n = 0, i;
  cl_device_id device = program->devices[device_i];
  llvm::Module **modptr = (llvm::Module **)&program->data[device_i];
  int error;
  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  llvm::Module *libmodule = getKernelLibrary(device, llvm_ctx);
  assert(libmodule != NULL);


  if (spir) {
#ifdef ENABLE_SPIR
    assert(num_input_programs == 1);
    POCL_RETURN_ERROR_ON ((device->endian_little == CL_FALSE),
                         CL_LINK_PROGRAM_FAILURE,
                         "SPIR is only supported on little-endian devices\n");

    concated_binaries.append((char *)cur_device_binaries[0],
                             cur_device_binary_sizes[0]);

    linked_module =
        parseModuleIRMem((char *)cur_device_binaries[0],
                         cur_device_binary_sizes[0], llvm_ctx->Context);

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
#ifndef __PPC64__
    if (linked_module->getModuleFlag("PIE Level") == nullptr)
      linked_module->setPIELevel(PIELevel::Large);
#endif

#else
    POCL_MSG_ERR("SPIR not supported\n");
    return CL_LINK_PROGRAM_FAILURE;
#endif
  } else {

    std::unique_ptr<llvm::Module> mod(
        new llvm::Module(StringRef("linked_program"), *llvm_ctx->Context));

    for (i = 0; i < num_input_programs; i++) {
      assert(cur_device_binaries[i]);
      assert(cur_device_binary_sizes[i]);
      concated_binaries.append((char *)cur_device_binaries[i],
                               cur_device_binary_sizes[i]);

      llvm::Module *p = (llvm::Module *)cur_llvm_irs[i];
      assert(p);

#ifdef LLVM_OLDER_THAN_7_0
      if (Linker::linkModules(*mod, llvm::CloneModule(p))) {
#else
      if (Linker::linkModules(*mod, llvm::CloneModule(*p))) {
#endif
        std::string msg = getDiagString(ctx);
        appendToProgramBuildLog(program, device_i, msg);
        return CL_LINK_PROGRAM_FAILURE;
      }
    }

    linked_module = mod.release();
  }

  if (linked_module == nullptr)
    return CL_LINK_PROGRAM_FAILURE;

  if (*modptr != nullptr) {
    delete *modptr;
    --llvm_ctx->number_of_IRs;
    *modptr = nullptr;
  }

  if (link_program) {
    // linked all the programs together, now link in the kernel library
    std::string log("Error(s) while linking: \n");
    if (link(linked_module, libmodule, log, device->global_as_id,
             device->device_aux_functions)) {
      appendToProgramBuildLog(program, device_i, log);
      std::string msg = getDiagString(ctx);
      appendToProgramBuildLog(program, device_i, msg);
      delete linked_module;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  *modptr = linked_module;
  ++llvm_ctx->number_of_IRs;

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

  /* Always retain program.bc for metadata */
  error = pocl_write_module(linked_module, program_bc_path, 0);
  if (error)
    return error;

  /* To avoid writing & reading the same back,
   * save program->binaries[i]
   */
  std::string content;
  writeModuleIRtoString(linked_module, content);

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
  const char *res = NULL;

  if (!llvm::sys::getHostCPUFeatures(Features)) {
    POCL_MSG_WARN ("getX86KernelLibName(): LLVM can't get host CPU flags!\n");
    /* getX86KernelLibName should only ever be enabled
       on x86-64, which always has sse2 */
    return "sse2";
  }

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


/**
 * Return the OpenCL C built-in function library bitcode
 * for the given device.
 */
static llvm::Module *getKernelLibrary(cl_device_id device,
                                      PoclLLVMContextData *llvm_ctx) {
  Triple triple(device->llvm_target_triplet);
  llvm::LLVMContext *llvmContext = llvm_ctx->Context;
  kernelLibraryMapTy *kernelLibraryMap = llvm_ctx->kernelLibraryMap;

  if (kernelLibraryMap->find(device) != kernelLibraryMap->end())
    return kernelLibraryMap->at(device);

  const char *subdir = "host";
  bool is_host = true;
  // TODO: move this to the device layer, a property to ask for
  // the kernel builtin bitcode library name, including its subdir
#if defined(TCE_AVAILABLE)
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
  kernellib = getPoclPrivateDataDir();
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
    if (device->llvm_cpu)
      kernellib += device->llvm_cpu;
#endif
  }
  kernellib += ".bc";

  llvm::Module *lib;

  if (pocl_exists(kernellib.c_str()))
    {
      POCL_MSG_PRINT_LLVM("Using %s as the built-in lib.\n", kernellib.c_str());
      lib = parseModuleIR(kernellib.c_str(), llvmContext);
    }
  else
    {
#ifndef KERNELLIB_HOST_DISTRO_VARIANTS
      if (is_host && pocl_exists(kernellib_fallback.c_str()))
        {
          POCL_MSG_WARN("Using fallback %s as the built-in lib.\n",
                        kernellib_fallback.c_str());
          lib = parseModuleIR(kernellib_fallback.c_str(), llvmContext);
        }
      else
#endif
        POCL_ABORT("Kernel library file %s doesn't exist.\n", kernellib.c_str());
    }
  assert (lib != NULL);
  kernelLibraryMap->insert(std::make_pair(device, lib));

  return lib;
}

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
