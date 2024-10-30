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

#include "clang/Lex/PreprocessorOptions.h"
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/LangOptions.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>

#include "llvm/LinkAllPasses.h"
#include "llvm/Linker/Linker.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>

#if LLVM_VERSION_MAJOR > 15
#include "llvm/TargetParser/Host.h"
#elif LLVM_VERSION_MAJOR > 10
#include "llvm/Support/Host.h"
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
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "LLVMUtils.h"
#include "pocl_util.h"

using namespace clang;
using namespace llvm;

POP_COMPILER_DIAGS

#include "ProgramScopeVariables.h"
#include "linker.h"

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

/**
 * Runs various LLVM "passes" on the program.bc LLVM module;
 * the passes are not real LLVM passes, but perhaps it will make sense
 * to convert at some point. Note that this should only run passes which
 * for some reason must be run at program.bc stage rather than parallel.bc
 *
 * \param [in] Context the pocl LLVM context
 * \param [in] Mod is the LLVM module
 * \param [in] Program the cl_program corresponding to Mod
 * \param [in] Device the device used for the LLVM passes
 * \param [in] device_i index into program->devices[] corresponding to Device
 * \param [out] Log a std::string containing the error/warning log
 * \returns true if there is an error
 *
 */
static bool generateProgramBC(PoclLLVMContextData *Context, llvm::Module *Mod,
                             cl_program Program, cl_device_id Device,
                             unsigned device_i, std::string &Log) {

  llvm::Module *BuiltinLib = getKernelLibrary(Device, Context);
  if (BuiltinLib == nullptr)
    return true;

  if (Device->run_program_scope_variables_pass) {
    size_t TotalGVarBytes = 0;
    if (runProgramScopeVariablesPass(Mod, Device->global_as_id,
                                     Device->local_as_id, TotalGVarBytes, Log))
      return true;
    Program->global_var_total_size[device_i] = TotalGVarBytes;
  }

  if (link(Mod, BuiltinLib, Log, Device))
    return true;

  raw_string_ostream OS(Log);
  bool BrokenDebugInfo = false;
  if (pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT)) {
    if (llvm::verifyModule(*Mod, &OS, &BrokenDebugInfo))
      return true;
  }

  if (BrokenDebugInfo)
    Log.append("Warning: broken DebugInfo detected\n");

  return false;
}

int pocl_llvm_build_program(cl_program program,
                            unsigned device_i,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names,
                            int linking_program)

{
  char tempfile[POCL_MAX_PATHNAME_LENGTH];
  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
  tempfile[0] = 0;
  llvm::Module *mod = nullptr;
  char temp_include_dir[POCL_MAX_PATHNAME_LENGTH];
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
      pocl_write_file(path.c_str(), input_header, input_header_size, 0);
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

#if !(defined(__x86_64__) && defined(__GNUC__))
  if (program->flush_denorms) {
    POCL_MSG_WARN("flush to zero is currently only implemented for "
                  "x86-64 & gcc/clang, ignoring flag\n");
  }
#endif

  /* temp dir takes preference */
  if (num_input_headers > 0)
    ss << "-I" << temp_include_dir << " ";

  if (device->has_64bit_long)
    ss << "-Dcl_khr_int64 ";

  if (!device->use_only_clang_opencl_headers) {
    ss << "-DPOCL_DEVICE_ADDRESS_BITS=" << device->address_bits << " ";
    ss << "-D__USE_CLANG_OPENCL_C_H ";
  }

  ss << "-xcl ";
  // Remove the inline keywords to force the user functions
  // to be included in the program. Otherwise they will
  // be removed and not inlined due to -O0.
  ss << "-Dinline= ";
  // The current directory is a standard search path.
  ss << "-I. ";
  // required for clGetKernelArgInfo()
  ss << "-cl-kernel-arg-info ";

#if (LLVM_MAJOR == 15) || (LLVM_MAJOR == 16)
#ifdef LLVM_OPAQUE_POINTERS
  ss << "-opaque-pointers ";
#else
  ss << "-no-opaque-pointers ";
#endif
#endif

  std::string fp_contract;
  if (device->llvm_fp_contract_mode != NULL) {
    fp_contract = std::string(device->llvm_fp_contract_mode);
  } else {
    fp_contract = "on";
  }

  size_t fastmath_flag = user_options.find("-cl-fast-relaxed-math");

  if (fastmath_flag != std::string::npos) {
#ifdef ENABLE_CONFORMANCE
    user_options.replace(fastmath_flag, 21,
                         "-cl-finite-math-only -cl-unsafe-math-optimizations");
#endif
    ss << "-D__FAST_RELAXED_MATH__=1 ";
    fp_contract = "fast";
  }

  size_t unsafemath_flag = user_options.find("-cl-unsafe-math-optimizations");

  if (unsafemath_flag != std::string::npos) {
#ifdef ENABLE_CONFORMANCE
    // this should be almost the same but disables -freciprocal-math.
    // required for conformance_math_divide test to pass with OpenCL 3.0
    user_options.replace(unsafemath_flag, 29,
                         "-cl-no-signed-zeros -cl-mad-enable -ffp-contract=fast");
#endif
    fp_contract = "fast";
  }

  ss << user_options << " ";

  if (device->endian_little)
    ss << "-D__ENDIAN_LITTLE__=1 ";

  if (device->image_support)
    ss << "-D__IMAGE_SUPPORT__=1 ";
  else {
    // workaround for a bug in Clang. It unconditionally predefines this macro
    // when compiling for SPIR or SPIRV target
    ss << "-U__IMAGE_SUPPORT__ ";
    ss << "-U__opencl_c_images ";
    ss << "-U__opencl_c_read_write_images ";
    ss << "-U__opencl_c_3d_image_writes ";
    // required for SPIR-V
    ss << "-D__undef___opencl_c_read_write_images ";
  }
  if (device->wg_collective_func_support == CL_FALSE)
    ss << "-D__undef___opencl_c_work_group_collective_functions ";
  if ((device->atomic_memory_capabilities &
       CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) == 0)
    ss << "-D__undef___opencl_c_atomic_scope_all_devices ";

  ss << "-DCL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" << device->global_var_max_size << " ";

  ss << "-D__OPENCL_VERSION__=" << device->version_as_int << " ";

  if (user_options.find("-cl-std=") == std::string::npos)
    ss << "-cl-std=" << device->opencl_c_version_as_opt << " ";

  std::string temp(ss.str());
  size_t pos = temp.find("-cl-std=CL");
  pos += 10;
  int cl_std_major = temp.c_str()[pos] - '0';
  int cl_std_minor = temp.c_str()[pos+2] - '0';
  int cl_std_i = cl_std_major * 100 + cl_std_minor * 10;
  ss << "-D__OPENCL_C_VERSION__=" << cl_std_i << " ";

  std::string exts = device->extensions;
  if (cl_std_major >= 3 && device->features != nullptr) {
    exts += ' ';
    exts += device->features;
  }
  llvm::StringRef extensions(exts);

  std::string cl_ext;
  if (extensions.size() > 0) {
    size_t e_start = 0, e_end = 0;
    while (e_end < std::string::npos) {
      while (e_start < extensions.size() && std::isspace(extensions[e_start]))
        ++e_start;
      if (e_start >= extensions.size())
        break;
      e_end = extensions.find(' ', e_start);
      if (e_end > extensions.size())
        e_end = extensions.size();
      llvm::StringRef tok = extensions.slice(e_start, e_end);
      e_start = e_end + 1;
      ss << "-D" << tok.str() << "=1 ";
      cl_ext += "+";
      cl_ext += tok.str();
      cl_ext += ",";
    }
    if (device->image_support == CL_FALSE)  {
      cl_ext += "-__opencl_c_images,";
      cl_ext += "-__opencl_c_read_write_images,";
      cl_ext += "-__opencl_c_3d_image_writes,";
    }
  }
  if (!cl_ext.empty()) {
    cl_ext.back() = ' '; // replace last "," with space
    ss << "-cl-ext=-all," << cl_ext;
  }

  // do not use LLVM builtin functions, rely on PoCL bitcode library only
  ss << "-fno-builtin ";
  // do not use jump/switch tables, these create a problem for VUA pass
  ss << "-fno-jump-tables ";

  // This is required otherwise the initialization fails with
  // unknown triple ''
  ss << "-triple=" << device->llvm_target_triplet << " ";
  if (device->llvm_cpu != NULL)
    ss << "-target-cpu " << device->llvm_cpu << " ";
  if (device->llvm_abi != NULL)
    ss << "-target-abi " << device->llvm_abi << " ";

  std::string AllBuildOpts = ss.str();

  POCL_MSG_PRINT_LLVM("all build options: %s\n", AllBuildOpts.c_str());

  char WSReplacementChar = 0;

  char *TempOptions = (char *)malloc(AllBuildOpts.length() + 1);

  memset(TempOptions, 0, AllBuildOpts.length() + 1);
  strncpy(TempOptions, AllBuildOpts.c_str(), AllBuildOpts.length());

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

    // Remove the quotes, Clang doesn't parse them. Note: cannot
    // replace with whitespace as Clang (at least v15) gets confused by
    // the whitespace (perhaps thinks it's part of the dir).
    s = std::regex_replace(s, std::regex("\""), "");

    // Clang (at least v15) gets confused if there's space after -I and
    // silently fails to add the include directory. Remove the space.
    // There can be space even without quotes in the user input.
    s = std::regex_replace(s, std::regex("-I(\\s+)"), "-I");

    // Convert the -g to more specific debug options understood by CFE.
    if (s == "-g") {
      itemstrs.push_back("-debug-info-kind=limited");
      itemstrs.push_back("-dwarf-version=4");
      itemstrs.push_back("-debugger-tuning=gdb");
      continue;
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
  std::cerr << "### options: " << AllBuildOpts << std::endl
            << std::endl
            << "user_options: " << user_options << std::endl
            << std::endl
            << "c_strs: ";
  for (auto cstr : itemcstrs) {
    std::cerr << cstr << " ";
  }
  std::cerr << std::endl;
#endif

  if (!CompilerInvocation::CreateFromArgs(
          pocl_build,
          ArrayRef<const char *>(itemcstrs.data(),
                                 itemcstrs.data() + itemcstrs.size()),
          diags)) {
    pocl_cache_create_program_cachedir(program, device_i, program->source,
                                       strlen(program->source),
                                       program_bc_path);
    get_build_log(program, device_i, ss_build_log, diagsBuffer,
                  CI.hasSourceManager() ? &CI.getSourceManager() : nullptr);
    return CL_INVALID_BUILD_OPTIONS;
  }

#if LLVM_MAJOR < 18
  LangOptions *la = pocl_build.getLangOpts();
#else
  LangOptions L = pocl_build.getLangOpts();
  LangOptions *la = &L;
#endif
  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();
  llvm::Triple triple (device->llvm_target_triplet);

#if LLVM_MAJOR >= 15
  LangOptions::setLangDefaults(*la, clang::Language::OpenCL, triple,
                               po.Includes, clang::LangStandard::lang_opencl12);
#else
  pocl_build.setLangDefaults(*la,
                             clang::InputKind(clang::Language::OpenCL),
                             triple,
                             po.Includes,
                             clang::LangStandard::lang_opencl12);
#endif

  // LLVM 3.3 and older do not set that char is signed which is
  // defined by the OpenCL C specs (but not by C specs).
  la->CharIsSigned = true;

  // the per-file types don't seem to override this
  la->OpenCLVersion = cl_std_i;
  la->FakeAddressSpaceMap = false;
  la->Blocks = true; //-fblocks
  la->MathErrno = false; // -fno-math-errno
  la->NoBuiltin = true;  // -fno-builtin
  la->Freestanding = true; // -ffree-standing
  la->AsmBlocks = true;  // -fasm (?)

  // setLangDefaults overrides to FPM_On for OpenCL.
  // So, we need to manually set it after
  if (fp_contract == "fast") {
    la->setDefaultFPContractMode(LangOptions::FPM_Fast);
  } else if (fp_contract == "on") {
    la->setDefaultFPContractMode(LangOptions::FPM_On);
  } else if (fp_contract == "off") {
    la->setDefaultFPContractMode(LangOptions::FPM_Off);
  }

  la->setStackProtector(LangOptions::StackProtectorMode::SSPOff);

  la->PICLevel = PICLevel::BigPIC;
  la->PIE = 0;

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
    char temp[POCL_MAX_PATHNAME_LENGTH];
    pocl_get_private_datadir(temp);
    IncludeRoot = temp;
  }
  KernelH = IncludeRoot + "/include/_kernel.h";
  BuiltinRenamesH = IncludeRoot + "/include/_builtin_renames.h";
  PoclTypesH = IncludeRoot + "/include/pocl_types.h";

  if (device->use_only_clang_opencl_headers == CL_FALSE) {
    po.Includes.push_back(PoclTypesH);
    po.Includes.push_back(BuiltinRenamesH);
  }
  // Use Clang's opencl-c.h header.
  po.Includes.push_back(IncludeRoot + "/include/opencl-c-base.h");
  po.Includes.push_back(IncludeRoot + "/include/opencl-c.h");

  if (device->use_only_clang_opencl_headers) {
    po.Includes.push_back(IncludeRoot + "/include/_clang_opencl.h");
  } else {
    po.Includes.push_back(KernelH);
  }
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
  char source_file[POCL_MAX_PATHNAME_LENGTH];
  POCL_RETURN_ERROR_ON(pocl_cache_write_program_source(source_file, program),
                       CL_OUT_OF_HOST_MEMORY, "Could not write program source");
  fe.Inputs.push_back(
      FrontendInputFile(source_file,
                        clang::InputKind(clang::Language::OpenCL)
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

    mod = (llvm::Module *)program->llvm_irs[device_i];
    if (mod != nullptr) {
      delete mod;
      program->llvm_irs[device_i] = nullptr;
      --llvm_ctx->number_of_IRs;
    }

    program->llvm_irs[device_i] = mod =
        parseModuleIR(program_bc_path, llvm_ctx->Context);
    assert(mod);
    ++llvm_ctx->number_of_IRs;

    parseModuleGVarSize(program, device_i, mod);

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

  mod = (llvm::Module *)program->llvm_irs[device_i];
  if (mod != nullptr) {
    delete mod;
    program->llvm_irs[device_i] = nullptr;
    --llvm_ctx->number_of_IRs;
  }

  mod = EmitLLVM.takeModule().release();
  if (mod == nullptr)
    return CL_BUILD_PROGRAM_FAILURE;
  else
    ++llvm_ctx->number_of_IRs;

  if (mod->getModuleFlag("PIC Level") == nullptr)
    mod->setPICLevel(PICLevel::BigPIC);

  // link w kernel lib, but not if we're called from clCompileProgram()
  // Later this should be replaced with indexed linking of source code
  // and/or bitcode for each kernel.
  if (linking_program) {
    std::string log("Error(s) while linking: \n");
    if (generateProgramBC(llvm_ctx, mod, program, device, device_i, log)) {
      appendToProgramBuildLog(program, device_i, log);
      std::string msg = getDiagString(ctx);
      appendToProgramBuildLog(program, device_i, msg);
      delete mod;
      mod = nullptr;
      --llvm_ctx->number_of_IRs;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  program->llvm_irs[device_i] = mod;

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc */
  error = pocl_write_module(mod, program_bc_path);
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

/* converts a "spir-unknown-unknown" module triple to target (CPU/CUDA etc)
 * triple, resets the datalayout to the target datalayout */
static int pocl_convert_spir_bitcode_to_target(llvm::Module *p,
                                               llvm::Module *libmodule,
                                               cl_device_id device) {
  const std::string &ModTriple = p->getTargetTriple();
  if (ModTriple.find("spir") == 0) {
    POCL_RETURN_ERROR_ON((device->endian_little == CL_FALSE),
                         CL_LINK_PROGRAM_FAILURE,
                         "SPIR is only supported on little-endian devices\n");
    size_t SpirAddrBits = Triple(ModTriple).isArch64Bit() ? 64 : 32;

    if (device->address_bits != SpirAddrBits) {
      delete p;
      POCL_RETURN_ERROR_ON(1, CL_LINK_PROGRAM_FAILURE,
                           "Device address bits != SPIR binary triple address "
                           "bits, device: %s / module: %s\n",
                           device->llvm_target_triplet, ModTriple.c_str());
    }

    /* Note this is a hack to get SPIR working. We'll be linking the
     * host kernel library (plain LLVM IR) to the SPIR program.bc,
     * so LLVM complains about incompatible DataLayouts.
     */
    p->setTargetTriple(libmodule->getTargetTriple());
    p->setDataLayout(libmodule->getDataLayout());

    if (p->getModuleFlag("PIC Level") == nullptr)
      p->setPICLevel(PICLevel::BigPIC);
    return CL_SUCCESS;
  }
  return CL_SUCCESS;
}

int pocl_llvm_link_program(cl_program program, unsigned device_i,
                           cl_uint num_input_programs,
                           unsigned char **cur_device_binaries,
                           size_t *cur_device_binary_sizes, void **cur_llvm_irs,
                           int link_device_builtin_library,
                           int linking_into_new_cl_program) {

  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
  std::string concated_binaries;
  size_t n = 0, i;
  cl_device_id device = program->devices[device_i];
  llvm::Module **modptr = (llvm::Module **)&program->llvm_irs[device_i];
  int error;
  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  llvm::Module *LibraryModule = getKernelLibrary(device, llvm_ctx);
  if (LibraryModule == nullptr)
    return CL_LINK_PROGRAM_FAILURE;

  std::unique_ptr<llvm::Module> mod(
      new llvm::Module(StringRef("linked_program"), *llvm_ctx->Context));
  llvm::Module *LinkedModule = nullptr;
  std::unique_ptr<llvm::Module> TempModule;
  mod->setTargetTriple(LibraryModule->getTargetTriple());
  mod->setDataLayout(LibraryModule->getDataLayout());
  mod->setPICLevel(PICLevel::BigPIC);


  // link the provided modules together into a single module
  for (i = 0; i < num_input_programs; i++) {
    assert(cur_device_binaries[i]);
    assert(cur_device_binary_sizes[i]);
    concated_binaries.append((char *)cur_device_binaries[i],
                             cur_device_binary_sizes[i]);

    if (cur_llvm_irs && cur_llvm_irs[i]) {
      llvm::Module *Ptr = (llvm::Module *)cur_llvm_irs[i];
      TempModule = llvm::CloneModule(*Ptr);
    } else {
      llvm::Module *Ptr =
          parseModuleIRMem((char *)cur_device_binaries[i],
                           cur_device_binary_sizes[i], llvm_ctx->Context);
      POCL_RETURN_ERROR_ON((Ptr == nullptr), CL_LINK_PROGRAM_FAILURE,
                           "could not parse module\n");
      TempModule.reset(Ptr);
    }

    error = pocl_convert_spir_bitcode_to_target(TempModule.get(), LibraryModule,
                                                device);
    POCL_RETURN_ERROR_ON((error != CL_SUCCESS), CL_LINK_PROGRAM_FAILURE,
                         "could not convert SPIR to Target\n");

    if (Linker::linkModules(*mod, std::move(TempModule))) {
      std::string msg = getDiagString(ctx);
      appendToProgramBuildLog(program, device_i, msg);
      return CL_LINK_PROGRAM_FAILURE;
    }
  }

  LinkedModule = mod.release();
  if (LinkedModule == nullptr)
    return CL_LINK_PROGRAM_FAILURE;
  // delete previous build of program
  if (*modptr != nullptr) {
    delete *modptr;
    --llvm_ctx->number_of_IRs;
    *modptr = nullptr;
  }

  // link the builtin library
  if (link_device_builtin_library) {
    // linked all the programs together, now link in the kernel library
    std::string log("Error(s) while linking: \n");
    if (generateProgramBC(llvm_ctx, LinkedModule, program, device, device_i,
                          log)) {
      appendToProgramBuildLog(program, device_i, log);
      std::string msg = getDiagString(ctx);
      appendToProgramBuildLog(program, device_i, msg);
      delete LinkedModule;
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }

  *modptr = LinkedModule;
  ++llvm_ctx->number_of_IRs;

  /* if we're linking binaries into a new cl_program, create cache
   * on concated binary contents (in undefined order); this is not
   * terribly useful, but we have to store it somewhere.. */
  if (linking_into_new_cl_program) {
    // assert build_hash is empty
    unsigned bhash_valid = pocl_cache_buildhash_is_valid (program, device_i);
    assert (!bhash_valid);
    error = pocl_cache_create_program_cachedir(
        program, device_i, concated_binaries.c_str(), concated_binaries.size(),
        program_bc_path);
    if (error) {
      POCL_MSG_ERR("pocl_cache_create_program_cachedir(%s)"
                   " failed with %i\n",
                   program_bc_path, error);
      return error;
    }
  } else {
    /* If we're linking existing cl_program, just get the path
     * assumes the program->build_hash[i] is already valid. */
    pocl_cache_program_bc_path(program_bc_path, program, device_i);
  }

  POCL_MSG_PRINT_LLVM("Writing program.bc to %s.\n", program_bc_path);

  /* Always retain program.bc for metadata */
  error = pocl_write_module(LinkedModule, program_bc_path);
  if (error)
    return error;

  /* To avoid writing & reading the same back, save program->binaries[i] */
  std::string content;
  writeModuleIRtoString(LinkedModule, content);

  if (program->binaries[device_i])
    POCL_MEM_FREE(program->binaries[device_i]);

  n = content.size();
  program->binary_sizes[device_i] = n;
  program->binaries[device_i] = (unsigned char *)malloc(n);
  std::memcpy(program->binaries[device_i], content.c_str(), n);

  return CL_SUCCESS;
}

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

  std::string BuiltinLibraryCommon, BuiltinLibrary, BuiltinLibraryFallback;

#ifdef ENABLE_POCL_BUILDING
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    BuiltinLibraryCommon = BUILDDIR;
    BuiltinLibraryCommon += "/lib/kernel/";
    BuiltinLibraryCommon += device->kernellib_subdir;
  } else // POCL_BUILDING == 0, use install dir
#endif
  {
    char temp[POCL_MAX_PATHNAME_LENGTH];
    pocl_get_private_datadir(temp);
    BuiltinLibraryCommon = temp;
  }

  BuiltinLibraryCommon += "/";

  BuiltinLibrary = BuiltinLibraryCommon + device->kernellib_name;
  BuiltinLibrary += ".bc";

  if (device->kernellib_fallback_name) {
    BuiltinLibraryFallback =
        BuiltinLibraryCommon + device->kernellib_fallback_name;
    BuiltinLibraryFallback += ".bc";
  }

  llvm::Module *BuiltinLibModule = nullptr;

  if (pocl_exists(BuiltinLibrary.c_str())) {
    POCL_MSG_PRINT_LLVM("Using %s as the built-in lib.\n",
                        BuiltinLibrary.c_str());
    BuiltinLibModule = parseModuleIR(BuiltinLibrary.c_str(), llvmContext);
  } else {
    if (device->kernellib_fallback_name &&
        pocl_exists(BuiltinLibraryFallback.c_str())) {
      POCL_MSG_WARN("Using fallback %s as the built-in lib.\n",
                    BuiltinLibraryFallback.c_str());
      BuiltinLibModule =
          parseModuleIR(BuiltinLibraryFallback.c_str(), llvmContext);
    } else
      POCL_MSG_ERR("Kernel library file %s doesn't exist.\n",
                   BuiltinLibrary.c_str());
  }
  if (BuiltinLibModule)
    kernelLibraryMap->insert(std::make_pair(device, BuiltinLibModule));

  return BuiltinLibModule;
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

  clang::driver::Driver TheDriver(pocl_get_path("CLANG", CLANG),
                                  Device->llvm_target_triplet, Diags);

  const char **ArgsEnd = Args;
  while (*ArgsEnd++ != nullptr) {}
  llvm::SmallVector<const char*, 0> ArgsArray(Args, ArgsEnd);

  int NumExtraArgs;
  const char *ExtraArgs = pocl_get_args("CLANG", &NumExtraArgs);
  const char *ExtraArg = ExtraArgs;
  for (int i = 0; i < NumExtraArgs; ++i) {
    ArgsArray.push_back(ExtraArg);
    ExtraArg += strlen(ExtraArg) + 1;
  }

  std::unique_ptr<clang::driver::Compilation> C(
      TheDriver.BuildCompilation(ArgsArray));

  free((void *)ExtraArgs);

  if (C && !C->containsError()) {
    SmallVector<std::pair<int, const clang::driver::Command *>, 4> FailingCommands;
    return TheDriver.ExecuteCompilation(*C, FailingCommands);
  } else {
    return -1;
  }

}
