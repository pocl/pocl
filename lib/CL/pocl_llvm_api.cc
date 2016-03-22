/* pocl_llvm_api.cc: C wrappers for calling the LLVM/Clang C++ APIs to invoke
   the different kernel compilation phases.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2015 Pekka Jääskeläinen
   
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

// For some reason including pocl.h before including CodeGenAction.h
// causes an error. Some kind of macro definition issue. To investigate.
#include "pocl.h"

#include "llvm/LinkAllPasses.h"
#ifdef LLVM_OLDER_THAN_3_7
#include "llvm/PassManager.h"
#include "llvm/Target/TargetLibraryInfo.h"
#else
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
using llvm::legacy::PassManager;
#endif

#include "llvm/Bitcode/ReaderWriter.h"
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

// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros, 
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm.h"
#include "pocl_runtime_config.h"
#include "install-paths.h"
#include "LLVMUtils.h"
#include "linker.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"

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
LLVMContext *GlobalContext() {
  if (globalContext == NULL) globalContext = new LLVMContext();
  return globalContext;
}

/* The LLVM API interface functions are not at the moment not thread safe,
   ensure only one thread is using this layer at the time with a mutex. */

static llvm::sys::Mutex kernelCompilerLock;

/* Global pocl device to be used by passes if needed */
cl_device_id currentPoclDevice = NULL;

static void InitializeLLVM();

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
    (FrontendInputFile(source_file, clang::IK_OpenCL));

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

#ifndef LLVM_OLDER_THAN_3_8
#define PassManager legacy::PassManager
#endif

static llvm::Module*
ParseIRFile(const char* fname, SMDiagnostic &Err, llvm::LLVMContext &ctx)
{
    return parseIRFile(fname, Err, ctx).release();
}

static void get_build_log(cl_program program,
                         unsigned device_i,
                         std::stringstream &ss_build_log,
                         clang::TextDiagnosticBuffer *diagsBuffer,
                         const SourceManager &sm)
{
    static const bool show_log = pocl_get_bool_option("POCL_VERBOSE", 0) ||
      pocl_get_bool_option("POCL_DEBUG", 0);

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

    pocl_cache_append_to_buildlog(program, device_i,
                                  ss_build_log.str().c_str(),
                                  ss_build_log.str().size());

    if (show_log)
      std::cerr << ss_build_log.str();

}


int pocl_llvm_build_program(cl_program program, 
                            unsigned device_i,
                            const char* user_options_cstr,
                            char* program_bc_path)

{
  void* write_lock = NULL;
  char tempfile[POCL_FILENAME_LENGTH];
  tempfile[0] = 0;
  llvm::Module **mod = NULL;
  std::string user_options(user_options_cstr ? user_options_cstr : "");
  std::string content;
  llvm::raw_string_ostream sos(content);
  size_t n = 0;

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

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

  if (extensions.size() > 0) {
    size_t e_start = 0, e_end = 0;
    while (e_end < std::string::npos) {
      e_end = extensions.find(' ', e_start);
      llvm::StringRef tok = extensions.slice(e_start, e_end);
      e_start = e_end + 1;
      ss << "-D" << tok.str() << " ";
    }
  }

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
  // if (cl_std_i != 10) && (cl_std != 11) && (cl_std != 12) (cl_std != 20)
  ss << "-D__OPENCL_C_VERSION__=" << cl_std_i << " ";

  /* With fp-contract we get calls to fma with processors which do not
      have fma instructions. These ruin the performance. Better to have
      the mul+add separated in the IR. */
  ss << "-fno-builtin -ffp-contract=off ";
  // This is required otherwise the initialization fails with
  // unknown triplet ''
  ss << "-triple=" << device->llvm_target_triplet << " ";
  if (device->llvm_cpu != NULL)
    ss << "-target-cpu " << device->llvm_cpu << " ";

#ifdef DEBUG_POCL_LLVM_API
  std::cout << "pocl_llvm_build_program: Final options: " << ss.str() << std::endl;
#endif

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
       diags))
    {
      pocl_cache_create_program_cachedir(program, device_i, NULL, 0,
        program_bc_path);
      get_build_log(program, device_i, ss_build_log, diagsBuffer, CI.getSourceManager());
      return CL_INVALID_BUILD_OPTIONS;
    }
  
  LangOptions *la = pocl_build.getLangOpts();
  pocl_build.setLangDefaults
    (*la, clang::IK_OpenCL, clang::LangStandard::lang_opencl12);
  
  // LLVM 3.3 and older do not set that char is signed which is
  // defined by the OpenCL C specs (but not by C specs).
  la->CharIsSigned = true;

  // the per-file types don't seem to override this 
  la->OpenCLVersion = cl_std_i;
  la->FakeAddressSpaceMap = true;
  la->Blocks = true; //-fblocks
  la->MathErrno = false; // -fno-math-errno
  la->NoBuiltin = true;  // -fno-builtin
  la->AsmBlocks = true;  // -fasm (?)

  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();

  std::string kernelh;
  if (pocl_get_bool_option("POCL_BUILDING", 0))
    { 
      kernelh  = SRCDIR;
      kernelh += "/include/_kernel.h";
    }
  else
    {
      kernelh = PKGDATADIR;
      kernelh += "/include/_kernel.h";
    }
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
  if (load_source(fe, program)!=0)
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
  cg.VectorizeBB = false;
  // This workarounds a Frontend codegen issues with an illegal address
  // space cast which is later flattened (and thus implicitly fixed) in
  // the TargetAddressSpaces. See:  https://github.com/pocl/pocl/issues/195
  cg.VerifyModule = false;

  PreprocessorOutputOptions &poo = pocl_build.getPreprocessorOutputOpts();
  poo.ShowCPP = 1;
  poo.ShowComments = 0;
  poo.ShowLineMarkers = 0;
  //poo.UseLineDirectives = 0;   // probably LLVM 3.7 thing
  poo.ShowMacroComments = 0;
  poo.ShowMacros = 1;
  poo.RewriteIncludes = 0;

  std::string saved_output(fe.OutputFile);
  pocl_cache_mk_temp_name(tempfile);
  fe.OutputFile = tempfile;

  bool success = true;
  clang::PrintPreprocessedAction action2;
  clang::EmitLLVMOnlyAction action(GlobalContext());
  success = CI.ExecuteAction(action2);
  if (!success)
    goto ERROR_BUILDLOG;

  char *preprocessed_out;
  uint64_t size;
  pocl_read_file(tempfile, &preprocessed_out, &size);
  pocl_remove(tempfile);
  fe.OutputFile = saved_output;

  if (!preprocessed_out)
    goto ERROR_BUILDLOG;

  pocl_cache_create_program_cachedir(program, device_i, preprocessed_out,
                                     size, program_bc_path);

  POCL_MEM_FREE(preprocessed_out);

  if (pocl_exists(program_bc_path)) {
    unlink_source(fe);
    return CL_SUCCESS;
  }

  // TODO: use pch: it is possible to disable the strict checking for
  // the compilation flags used to compile it and the current translation
  // unit via the preprocessor options directly.

  success = CI.ExecuteAction(action);

  unlink_source(fe);

  get_build_log(program, device_i, ss_build_log, diagsBuffer, CI.getSourceManager());

  if (!success)
    return CL_BUILD_PROGRAM_FAILURE;

  mod = (llvm::Module **)&program->llvm_irs[device_i];
  if (*mod != NULL)
    delete (llvm::Module*)*mod;

  *mod = action.takeModule().release();

  if (*mod == NULL)
    return CL_BUILD_PROGRAM_FAILURE;

  write_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
  assert(write_lock);

  /* Always retain program.bc. Its required in clBuildProgram */
  pocl_write_module(*mod, program_bc_path, 0);

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

ERROR_BUILDLOG:
  pocl_cache_create_program_cachedir(program, device_i, NULL, 0,
    program_bc_path);
  get_build_log(program, device_i, ss_build_log, diagsBuffer, CI.getSourceManager());
  return CL_BUILD_PROGRAM_FAILURE;

}

int pocl_llvm_get_kernel_arg_metadata(const char* kernel_name,
                                      llvm::Module *input,
                                      cl_kernel kernel)
{

  // find the right kernel in "opencl.kernels" metadata
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  llvm::MDNode *kernel_metadata = NULL;

  // Not sure what to do in this case
  if (!opencl_kernels || opencl_kernels->getNumOperands() == 0) return -1;

  for (unsigned i = 0, e = opencl_kernels->getNumOperands(); i != e; ++i) {
    llvm::MDNode *kernel_iter = opencl_kernels->getOperand(i);

    llvm::Function *kernel_prototype = 
      llvm::cast<llvm::Function>(
        dyn_cast<llvm::ValueAsMetadata>(kernel_iter->getOperand(0))->getValue());
    std::string name = kernel_prototype->getName().str();
    if (name == kernel_name) {
      kernel_metadata = kernel_iter;
      break;
    }
  }

  kernel->has_arg_metadata = 0;
  int bitcode_is_spir = input->getTargetTriple().find("spir") == 0;

  assert(kernel_metadata && "kernel NOT found in opencl.kernels metadata");

  unsigned e = kernel_metadata->getNumOperands();
  for (unsigned i = 1; i != e; ++i) {
    llvm::MDNode *meta_node = llvm::cast<MDNode>(kernel_metadata->getOperand(i));

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
        //std::cout << "with value: " << val << std::endl;
        if (bitcode_is_spir) {
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
            case POCL_ADDRESS_SPACE_PRIVATE:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
            case POCL_ADDRESS_SPACE_GLOBAL:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL; break;
            case POCL_ADDRESS_SPACE_LOCAL:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL; break;
            case POCL_ADDRESS_SPACE_CONSTANT:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT; break;
            case POCL_ADDRESS_SPACE_GENERIC:
              current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE; break;
          }
        }
      }
      else if (isa<MDString>(meta_node->getOperand(j))) {
        //std::cout << "is MDString" << std::endl;
        llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
        std::string val = m->getString().str();
        //std::cout << "with value: " << val << std::endl;
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
          current_arg->type_name = new char[val.size() + 1];
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
          current_arg->name = new char[val.size() + 1];
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

int pocl_llvm_get_kernel_metadata(cl_program program, 
                                  cl_kernel kernel,
                                  int device_i,     
                                  const char* kernel_name,
                                  int * errcode)
{

  int i;
  llvm::Module *input = NULL;
  assert(program->devices[device_i]->llvm_target_triplet && 
         "Device has no target triple set"); 

  if (program->llvm_irs != NULL &&
      program->llvm_irs[device_i] != NULL)
    {
      input = (llvm::Module*)program->llvm_irs[device_i];
#ifdef DEBUG_POCL_LLVM_API
      printf("### use a saved llvm::Module\n");
#endif
    }
  else
    {
      *errcode = CL_INVALID_PROGRAM_EXECUTABLE;
      return 1;
    }


#ifdef DEBUG_POCL_LLVM_API        
  printf("### fetching kernel metadata for kernel %s program %p input llvm::Module %p\n",
         kernel_name, program, input);
#endif

  llvm::Function *kernel_function = input->getFunction(kernel_name);
  if (!kernel_function) {
    *errcode = CL_INVALID_KERNEL_NAME;
    return 1;
  }

  DataLayout *TD = 0;
#ifdef LLVM_OLDER_THAN_3_7
  const std::string &ModuleDataLayout = input->getDataLayout()->getStringRepresentation();
#else
  const std::string &ModuleDataLayout = input->getDataLayout().getStringRepresentation();
#endif
  if (!ModuleDataLayout.empty())
    TD = new DataLayout(ModuleDataLayout);

  const llvm::Function::ArgumentListType &arglist = 
      kernel_function->getArgumentList();
  kernel->num_args = arglist.size();

  SmallVector<GlobalVariable *, 8> locals;
  for (llvm::Module::global_iterator i = input->global_begin(),
         e = input->global_end();
       i != e; ++i) {
    std::string funcName = "";
    funcName = kernel_function->getName().str();
    if (pocl::is_automatic_local(funcName, *i))
      {
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

  kernel->arg_info = (struct pocl_argument_info*) calloc(kernel->num_args, sizeof(struct pocl_argument_info));
  memset(kernel->arg_info, 0, sizeof(struct pocl_argument_info)*kernel->num_args);

  i = 0;
  for( llvm::Function::const_arg_iterator ii = arglist.begin(), 
                                          ee = arglist.end(); 
       ii != ee ; ii++)
  {
    llvm::Type *t = ii->getType();
    kernel->arg_info[i].type = POCL_ARG_TYPE_NONE;
    const llvm::PointerType *p = dyn_cast<llvm::PointerType>(t);
    if (p && !ii->hasByValAttr()) {
      kernel->arg_info[i].type = POCL_ARG_TYPE_POINTER;
      // index 0 is for function attributes, parameters start at 1.
      if (p->getAddressSpace() == POCL_ADDRESS_SPACE_GLOBAL ||
          p->getAddressSpace() == POCL_ADDRESS_SPACE_CONSTANT ||
          pocl::is_image_type(*t) || pocl::is_sampler_type(*t))
        {
          kernel->arg_info[i].is_local = false;
        }
      else
        {
          if (p->getAddressSpace() != POCL_ADDRESS_SPACE_LOCAL)
            {
              p->dump();
              assert(p->getAddressSpace() == POCL_ADDRESS_SPACE_LOCAL);
            }
          kernel->arg_info[i].is_local = true;
        }
    } else {
      kernel->arg_info[i].is_local = false;
    }
        
    if (pocl::is_image_type(*t))
      {
        kernel->arg_info[i].type = POCL_ARG_TYPE_IMAGE;
      } 
    else if (pocl::is_sampler_type(*t)) 
      {
        kernel->arg_info[i].type = POCL_ARG_TYPE_SAMPLER;
      }
    i++;  
  }
  // fill 'kernel->reqd_wg_size'
  kernel->reqd_wg_size = (int*)malloc(3*sizeof(int));

  unsigned reqdx = 0, reqdy = 0, reqdz = 0;

  llvm::NamedMDNode *size_info = 
    kernel_function->getParent()->getNamedMetadata("opencl.kernel_wg_size_info");
  if (size_info) {
    for (unsigned i = 0, e = size_info->getNumOperands(); i != e; ++i) {
      llvm::MDNode *KernelSizeInfo = size_info->getOperand(i);
      if (dyn_cast<ValueAsMetadata>(
        KernelSizeInfo->getOperand(0).get())->getValue() != kernel_function) 
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
  kernel->reqd_wg_size[0] = reqdx;
  kernel->reqd_wg_size[1] = reqdy;
  kernel->reqd_wg_size[2] = reqdz;
  
#ifndef POCL_ANDROID
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous devices which need
  // these definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now

  std::stringstream content;

  content <<  "\n #include <pocl_device.h>\n";
  content << "void _pocl_launcher_" << kernel_name << "_workgroup(void** args, struct pocl_context*);\n";
  content << "void _pocl_launcher_" << kernel_name << "_workgroup_fast(void** args, struct pocl_context*);\n";
  content << "__attribute__((address_space(3))) __kernel_metadata _" << kernel_name << "_md = {\n";
  content << "     \"" << kernel_name << "\", /* name */ \n";
  content << "     " << kernel->num_args << ", /* num_args */\n";
  content << "     " << kernel->num_locals << ", /* num_locals */\n";
  content << "     _pocl_launcher_" << kernel_name << "_workgroup_fast\n";
  content << " };\n";

  pocl_cache_write_descriptor(program, device_i,
                              kernel_name, content.str().c_str(),
                              content.str().size());
#endif

  if (pocl_llvm_get_kernel_arg_metadata(kernel_name, input, kernel)) {
    *errcode = CL_INVALID_KERNEL;
    return 1;
  };

  *errcode = CL_SUCCESS;
  return 0;
}

char* get_cpu_name() {
  StringRef r = llvm::sys::getHostCPUName();
  assert(r.size() > 0);
  char* cpu_name = (char*) malloc (r.size()+1);
  strncpy(cpu_name, r.data(), r.size());
  cpu_name[r.size()] = 0;
  return cpu_name;
}

/* helpers copied from LLVM opt START */

/* FIXME: these options should come from the cl_device, and
 * cl_program's options. */
static llvm::TargetOptions GetTargetOptions() {
  llvm::TargetOptions Options;
  Options.PositionIndependentExecutable = true;
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
const char* getX86KernelLibName() {
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


// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(cl_device_id device,
 const std::vector<std::string>& MAttrs=std::vector<std::string>()) {

  std::string Error;
  Triple TheTriple(device->llvm_target_triplet);

  std::string MCPU =  device->llvm_cpu ? device->llvm_cpu : "";

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
  
  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  TargetMachine* TM = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                                     MCPU, FeaturesStr, 
                                                     GetTargetOptions(),
                                                     Reloc::PIC_, 
                                                     CodeModel::Default,
                                                     CodeGenOpt::Aggressive);
  assert (TM != NULL && "llvm target has no targetMachine constructor"); 
  if (device->ops->init_target_machine)
    device->ops->init_target_machine(device->data, TM);

  return TM;
}
/* helpers copied from LLVM opt END */

static void InitializeLLVM() {
  
  static bool LLVMInitialized = false;
  if (LLVMInitialized) return;
  // We have not initialized any pass managers for any device yet.
  // Run the global LLVM pass initialization functions.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  LLVMInitialized = true;
}

/**
 * Prepare the kernel compiler passes.
 *
 * The passes are created only once per program run per device.
 * The returned pass manager should not be modified, only the Module
 * should be optimized using it.
 */
static PassManager& kernel_compiler_passes
(cl_device_id device, const std::string& module_data_layout)
{
  static std::map<cl_device_id, PassManager*> kernel_compiler_passes;

  bool SPMDDevice = device->spmd;

  currentPoclDevice = device;

  if (kernel_compiler_passes.find(device) != 
      kernel_compiler_passes.end())
    {
      return *kernel_compiler_passes[device];
    }
  
  Triple triple(device->llvm_target_triplet);

  PassRegistry &Registry = *PassRegistry::getPassRegistry();

  const bool first_initialization_call = kernel_compiler_passes.size() == 0;

  if (first_initialization_call) {
    // TODO: do this globally, and just once per program
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
  }

  // Scalarizer is in LLVM upstream since 3.4.
  const bool SCALARIZE = pocl_is_option_set("POCL_SCALARIZE_KERNELS");

# ifdef LLVM_OLDER_THAN_3_7
  StringMap<llvm::cl::Option*> opts;
  llvm::cl::getRegisteredOptions(opts);
# else
  StringMap<llvm::cl::Option *>& opts = llvm::cl::getRegisteredOptions();
# endif

  PassManager *Passes = new PassManager();

#ifdef LLVM_OLDER_THAN_3_7
  // Need to setup the target info for target specific passes. */
  TargetMachine *Machine = GetTargetMachine(device);

  // Add internal analysis passes from the target machine.
  if (Machine != NULL)
    Machine->addAnalysisPasses(*Passes);
#else 
  TargetMachine *Machine = GetTargetMachine(device);
  if (Machine != NULL)
    Passes->add(createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));
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
  passes.push_back("workitem-handler-chooser");
  passes.push_back("mem2reg");
  passes.push_back("domtree");
  passes.push_back("break-constgeps");
  if(device->autolocals_to_args)
	  passes.push_back("automatic-locals");
  passes.push_back("flatten");
  passes.push_back("always-inline");
  passes.push_back("globaldce");
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
  }
  passes.push_back("workgroup");
  passes.push_back("allocastoentry");
  passes.push_back("target-address-spaces");
  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running the
  // standard LLVM optimizations.
  passes.push_back("simplifycfg");
  //passes.push_back("print-module");

  const std::string wg_method = 
    pocl_get_string_option("POCL_WORK_GROUP_METHOD", "loopvec");

  if (wg_method == "loopvec")
    {

      if (SCALARIZE)
        {
          printf("SCALARIZE\n");
          passes.push_back("scalarizer");
        }

      if (kernel_compiler_passes.size() == 0) 
        {
          // Set the options only once. TODO: fix it so that each
          // device can reset their own options. Now one cannot compile
          // with different options to different devices at one run.

          // LLVM inner loop vectorizer does not check whether the loop inside 
          // another loop, in which case even a small trip count loops might be 
          // worthwhile to vectorize.
          llvm::cl::Option *O = opts["vectorizer-min-trip-count"];
          assert(O && "could not find LLVM option 'vectorizer-min-trip-count'");
          O->addOccurrence(1, StringRef("vectorizer-min-trip-count"), StringRef("2"), false); 

          if (SCALARIZE) 
            {
              O = opts["scalarize-load-store"];
              assert(O && "could not find LLVM option 'scalarize-load-store'");
              O->addOccurrence(1, StringRef("scalarize-load-store"), StringRef(""), false); 
            }

#ifdef DEBUG_POCL_LLVM_API        
          printf ("### autovectorizer enabled\n");

          O = opts["debug-only"];
          assert(O && "could not find LLVM option 'debug'");
          O->addOccurrence(1, StringRef("debug-only"), StringRef("loop-vectorize"), false); 
#endif

          if (pocl_get_bool_option("POCL_VECTORIZER_REMARKS", 0) == 1) {
            // Enable diagnostics from the loop vectorizer.
            O = opts["pass-remarks-missed"];
            assert(O && "could not find LLVM option 'pass-remarks-missed'");
            O->addOccurrence(1, StringRef("pass-remarks-missed"), StringRef("loop-vectorize"), 
                             false); 

            O = opts["pass-remarks-analysis"];
            assert(O && "could not find LLVM option 'pass-remarks-analysis'");
            O->addOccurrence(1, StringRef("pass-remarks-analysis"), StringRef("loop-vectorize"), 
                             false); 

            O = opts["pass-remarks"];
            assert(O && "could not find LLVM option 'pass-remarks'");
            O->addOccurrence(1, StringRef("pass-remarks"), StringRef("loop-vectorize"), 
                             false); 
          }

          O = opts["unroll-threshold"];
          assert(O && "could not find LLVM option 'unroll-threshold'");
          O->addOccurrence(1, StringRef("unroll-threshold"), StringRef("1"), false); 
        }
    } 

  passes.push_back("instcombine");
  passes.push_back("STANDARD_OPTS");
  passes.push_back("instcombine");

  // Now actually add the listed passes to the PassManager.
  for(unsigned i = 0; i < passes.size(); ++i)
    {
    
      // This is (more or less) -O3
      if (passes[i] == "STANDARD_OPTS")
        {
          PassManagerBuilder Builder;
          Builder.OptLevel = 3;
          Builder.SizeLevel = 0;

          // These need to be setup in addition to invoking the passes
          // to get the vectorizers initialized properly.
          if (wg_method == "loopvec") {
            Builder.LoopVectorize = true;
            Builder.SLPVectorize = true;
#ifdef LLVM_OLDER_THAN_3_7
            Builder.BBVectorize = pocl_get_bool_option ("POCL_BBVECTORIZE", 1);
#else
            // In LLVM 3.7 the BB vectorizer crashes with some of the
            // the shuffle tests, but gives performance improvements in
            // some (see https://github.com/pocl/pocl/issues/251).
            // Disable by default because of
            // https://llvm.org/bugs/show_bug.cgi?id=25077
            Builder.BBVectorize = pocl_get_bool_option ("POCL_BBVECTORIZE", 0);
#endif
          }

          Builder.populateModulePassManager(*Passes);
     
          continue;
        }

      const PassInfo *PIs = Registry.getPassInfo(StringRef(passes[i]));
      if(PIs)
        {
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
  kernel_compiler_passes[device] = Passes;
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
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  static std::map<cl_device_id, llvm::Module*> libs;

  Triple triple(device->llvm_target_triplet);

  if (libs.find(device) != libs.end())
    return libs[device];

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
#ifdef POCL_BUILT_WITH_CMAKE
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
#endif
    }
  } else { // POCL_BUILDING == 0, use install dir
    kernellib = PKGDATADIR;
    kernellib += "/kernel-";
    kernellib += device->llvm_target_triplet;
    if (is_host) {
#ifdef POCL_BUILT_WITH_CMAKE
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
#endif
    }
  }
  kernellib += ".bc";

  llvm::Module *lib;
  SMDiagnostic Err;

  if (pocl_exists(kernellib.c_str()))
    {
      POCL_MSG_PRINT_INFO("Using %s as the built-in lib.\n", kernellib.c_str());
      lib = ParseIRFile(kernellib.c_str(), Err, *GlobalContext());
    }
  else
    {
      if (is_host && pocl_exists(kernellib_fallback.c_str()))
        {
          POCL_MSG_WARN("Using fallback %s as the built-in lib.\n",
                        kernellib_fallback.c_str());
          lib = ParseIRFile(kernellib_fallback.c_str(), Err, *GlobalContext());
        }
      else
        POCL_ABORT("Kernel library file %s doesn't exist.", kernellib.c_str());
    }
  assert (lib != NULL);
  libs[device] = lib;

  return lib;
}

/* This is used to control the kernel we want to process in the kernel compilation. */
extern cl::opt<std::string> KernelName;

int pocl_llvm_generate_workgroup_function(cl_device_id device, cl_kernel kernel,
                                          size_t local_x, size_t local_y, size_t local_z)
{

  pocl::WGDynamicLocalSize = (local_x == 0 && local_y == 0 && local_z == 0);

  cl_program program = kernel->program;
  int device_i = pocl_cl_device_to_index(program, device);
  assert(device_i >= 0);

  char parallel_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_work_group_function_path(parallel_bc_path, program, device_i, kernel, local_x, local_y, local_z);

  if (pocl_exists(parallel_bc_path))
    return CL_SUCCESS;

  char final_binary_path[POCL_FILENAME_LENGTH];
  pocl_cache_final_binary_path(final_binary_path, program, device_i, kernel, local_x, local_y, local_z);

  if (pocl_exists(final_binary_path))
    return CL_SUCCESS;

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

#ifdef DEBUG_POCL_LLVM_API        
  printf("### calling the kernel compiler for kernel %s local_x %zu "
         "local_y %zu local_z %zu parallel_filename: %s\n",
         kernel->name, local_x, local_y, local_z, parallel_bc_path);
#endif

  Triple triple(device->llvm_target_triplet);

  SMDiagnostic Err;
  std::string errmsg;

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
      pocl_cache_program_bc_path(program_bc_path, program, device_i);
      input = ParseIRFile(program_bc_path, Err, *GlobalContext());
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

  // Later this should be replaced with indexed linking of source code
  // and/or bitcode for each kernel.
  llvm::Module *libmodule = kernel_library(device);
  assert (libmodule != NULL);
  link(input, libmodule);

  /* Now finally run the set of passes assembled above */
  // TODO pass these as parameters instead, this is not thread safe!
  pocl::WGLocalSizeX = local_x;
  pocl::WGLocalSizeY = local_y;
  pocl::WGLocalSizeZ = local_z;
  KernelName = kernel->name;

#ifdef LLVM_OLDER_THAN_3_7
  kernel_compiler_passes(
      device,
      input->getDataLayout()->getStringRepresentation()).run(*input);
#else
  kernel_compiler_passes(
      device,
      input->getDataLayout().getStringRepresentation())
      .run(*input);
#endif
  // TODO: don't write this once LLC is called via API, not system()
  pocl_cache_write_kernel_parallel_bc(input, program, device_i, kernel,
                                  local_x, local_y, local_z);

  delete input;
  return 0;
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

  program->llvm_irs[device_i] =
              ParseIRFile(program_bc_path, Err, *GlobalContext());
  return 0;
}

void pocl_free_llvm_irs(cl_program program, int device_i)
{
    if (program->llvm_irs[device_i]) {
        llvm::Module *mod = (llvm::Module *)program->llvm_irs[device_i];
        delete mod;
        program->llvm_irs[device_i] = NULL;
    }
}

void pocl_llvm_update_binaries (cl_program program) {

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();
  char program_bc_path[POCL_FILENAME_LENGTH];
  void* cache_lock = NULL;

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

      pocl_cache_program_bc_path(program_bc_path, program, i);
      pocl_write_module((llvm::Module*)program->llvm_irs[i], program_bc_path, 1);

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
#ifdef DEBUG_POCL_LLVM_API        
      printf("### binary for device %zi was of size %zu\n", i, program->binary_sizes[i]);
#endif

    }
}

/* This is the implementation of the public pocl_llvm_get_kernel_count(),
 * and is used internally also by pocl_llvm_get_kernel_names to
 */
unsigned
pocl_llvm_get_kernel_count(cl_program program, llvm::NamedMDNode **md_ret)
{
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  // TODO: is it safe to assume every device (i.e. the index 0 here)
  // has the same set of programs & kernels?
  llvm::Module *mod = (llvm::Module *) program->llvm_irs[0];
  llvm::NamedMDNode *md = mod->getNamedMetadata("opencl.kernels");

  if (md_ret)
    *md_ret = md;

  if (md == NULL)
    return 0;

  return md->getNumOperands();
}

unsigned
pocl_llvm_get_kernel_count(cl_program program)
{
  return pocl_llvm_get_kernel_count(program, NULL);
}

unsigned
pocl_llvm_get_kernel_names( cl_program program, char **knames, unsigned max_num_krn )
{
  llvm::NamedMDNode *md;
  unsigned i, n = pocl_llvm_get_kernel_count(program, &md);

  for (i=0; i<n; i++) {
    assert( md->getOperand(i)->getOperand(0) != NULL);
    llvm::Function *k = 
      cast<Function>(
        dyn_cast<llvm::ValueAsMetadata>(md->getOperand(i)->getOperand(0))->getValue());
    if (i < max_num_krn)
      knames[i] = strdup(k->getName().data());
  }
  return i;
}

/* Run LLVM codegen on input file (parallel-optimized).
 *
 * Output native object file. */
int
pocl_llvm_codegen(cl_kernel kernel,
                  cl_device_id device,
                  const char *infilename,
                  const char *outfilename)
{
    llvm::MutexGuard lockHolder(kernelCompilerLock);

    SMDiagnostic Err;

    if (pocl_exists(outfilename))
      return 0;

    llvm::Triple triple(device->llvm_target_triplet);
    llvm::TargetMachine *target = GetTargetMachine(device);

    llvm::Module *input = ParseIRFile(infilename, Err, *GlobalContext());

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
    return pocl_write_file(outfilename, o.c_str(), o.size(), 0, 0);

}
/* vim: set ts=4 expandtab: */
