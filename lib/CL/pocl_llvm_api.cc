/* pocl_llvm_api.cc: C wrappers for calling the LLVM/Clang C++ APIs to invoke
   the different kernel compilation phases.

   Copyright (c) 2013 Kalle Raiskila 
                 2013-2014 Pekka Jääskeläinen
   
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

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/PassManager.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Transforms/Utils/Cloning.h"

#if (defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4)
#include "llvm/Linker.h"
#else
#include "llvm/Linker/Linker.h"
#include "llvm/PassAnalysisSupport.h"
#endif

#ifdef LLVM_3_2
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/IRReader.h"
#include "llvm/DataLayout.h"
#else
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IRReader/IRReader.h"
#endif

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
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

#ifndef _MSC_VER
#  include <unistd.h>
#endif

// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros, 
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm.h"
#include "pocl_runtime_config.h"
#include "install-paths.h"
#include "LLVMUtils.h"
#include "linker.h"
#include "pocl_util.h"

using namespace clang;
using namespace llvm;

#if defined LLVM_3_2 || defined LLVM_3_3
#include "llvm/Support/raw_ostream.h"
#define F_Binary llvm::raw_fd_ostream::F_Binary
#elif defined LLVM_3_4
using llvm::sys::fs::F_Binary;
#else
// a binary file is "not a text file"
#define F_Binary llvm::sys::fs::F_None
#endif


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

static void InitializeLLVM();

//#define DEBUG_POCL_LLVM_API

#if defined(DEBUG_POCL_LLVM_API) && defined(NDEBUG)
#undef NDEBUG
#include <cassert>
#endif

// Write a kernel compilation intermediate result
// to file on disk, if user has requested with environment
// variable
// TODO: what to do on errors?
static inline void
write_temporary_file( const llvm::Module *mod,
                      const char *filename )
{
  tool_output_file *Out;
  #if LLVM_VERSION_MAJOR==3 && LLVM_VERSION_MINOR<6
  std::string ErrorInfo;
  Out = new tool_output_file(filename, ErrorInfo, F_Binary);
  #else
  std::error_code ErrorInfo;
  Out = new tool_output_file(filename, ErrorInfo, F_Binary);
  #endif
  WriteBitcodeToFile(mod, Out->os());
  Out->keep();
  delete Out;
}

// Read input source to clang::FrontendOptions.
// The source is contained in the program->source array,
// but if debugging option is enabled in the kernel compiler
// we need to dump the file to disk first for the debugger
// to find it.
static inline int
load_source(FrontendOptions &fe,
            const char* temp_dir,
            cl_program program)
{
  std::string kernel_file(temp_dir);
  kernel_file += "/" POCL_PROGRAM_CL_FILENAME;
  std::ofstream ofs(kernel_file.c_str());
  ofs << program->source;
  if (!ofs.good())
    return CL_OUT_OF_HOST_MEMORY;
  fe.Inputs.push_back
    (FrontendInputFile(kernel_file, clang::IK_OpenCL));

  return 0;
}

// Compatibility function: this function existed up to LLVM 3.5
// With 3.6 its name & signature changed
#if !(defined LLVM_3_2 || defined LLVM_3_3 || \
      defined LLVM_3_4 || defined LLVM_3_5)
static llvm::Module*
ParseIRFile(const char* fname, SMDiagnostic &Err, llvm::LLVMContext &ctx)
{
    return parseIRFile(fname, Err, ctx).release();
}
#endif

int pocl_llvm_build_program(cl_program program, 
                            cl_device_id device, 
                            int device_i,     
                            const char* temp_dir,
                            const char* binary_file_name,
                            const char* device_tmpdir,
                            const char* user_options)

{
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

  // add device specific switches, if any
  std::stringstream ss;
  std::stringstream ss_build_log;

  std::stringstream build_log_filename;
  build_log_filename << temp_dir << "/" << POCL_BUILDLOG_FILENAME;
  /* Overwrite build log */
  std::ofstream fp(build_log_filename.str().c_str(), std::ofstream::trunc);
  fp.close();

  if (device->ops->init_build != NULL) 
    {
      assert (device_tmpdir != NULL);
      char *device_switches = 
        device->ops->init_build (device->data, device_tmpdir);
      if (device_switches != NULL) 
        {
          ss << device_switches << " ";
        }
      POCL_MEM_FREE(device_switches);
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

   /* With fp-contract we get calls to fma with processors which do not
      have fma instructions. These ruin the performance. Better to have
      the mul+add separated in the IR. */
  ss << "-fno-builtin -ffp-contract=off ";

  // This is required otherwise the initialization fails with
  // unknown triplet ''
  ss << "-triple=" << device->llvm_target_triplet << " ";
  if (device->llvm_cpu != NULL)
    ss << "-target-cpu " << device->llvm_cpu << " ";
  ss << user_options << " ";
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

  if (!CompilerInvocation::CreateFromArgs
      (pocl_build, itemcstrs.data(), itemcstrs.data() + itemcstrs.size(), 
       diags)) 
    {
      for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->err_begin(), 
             e = diagsBuffer->err_end(); i != e; ++i) 
        {
          ss_build_log << "error: " << (*i).second << std::endl;
        }
      for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->warn_begin(), 
             e = diagsBuffer->warn_end(); i != e; ++i) 
        {
          ss_build_log << "warning: " << (*i).second << std::endl;
        }
      pocl_create_or_append_file(build_log_filename.str().c_str(),
                                      ss_build_log.str().c_str());
      std::cerr << ss_build_log.str();
      return CL_INVALID_BUILD_OPTIONS;
    }
  
  LangOptions *la = pocl_build.getLangOpts();
  pocl_build.setLangDefaults
    (*la, clang::IK_OpenCL, clang::LangStandard::lang_opencl12);
  
  // LLVM 3.3 and older do not set that char is signed which is
  // defined by the OpenCL C specs (but not by C specs).
  la->CharIsSigned = true;

  // the per-file types don't seem to override this 
  la->OpenCLVersion = 120;
  la->FakeAddressSpaceMap = true;
  la->Blocks = true; //-fblocks
  la->MathErrno = false; // -fno-math-errno
  la->NoBuiltin = true;  // -fno-builtin
#ifndef LLVM_3_2
  la->AsmBlocks = true;  // -fasm (?)
#endif

  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();
  /* configure.ac sets a a few host specific flags for pthreads and
     basic devices. */
  if (device->has_64bit_long == 0)
    po.addMacroDef("_CL_DISABLE_LONG");

  po.addMacroDef("__OPENCL_VERSION__=120"); // -D__OPENCL_VERSION_=120

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

  // TODO: user_options (clBuildProgram options) are not passed

  clang::TargetOptions &ta = pocl_build.getTargetOpts();
  ta.Triple = device->llvm_target_triplet;
  if (device->llvm_cpu != NULL)
    ta.CPU = device->llvm_cpu;

  // printf("### Triple: %s, CPU: %s\n", ta.Triple.c_str(), ta.CPU.c_str());

  // FIXME: print out any diagnostics to stdout for now. These should go to a buffer for the user
  // to dig out. (and probably to stdout too, overridable with environment variables) 
#ifdef LLVM_3_2
  CI.createDiagnostics(0, NULL, diagsBuffer, false);
#else
  CI.createDiagnostics(diagsBuffer, false);
#endif 
 
  FrontendOptions &fe = pocl_build.getFrontendOpts();
  // The CreateFromArgs created an stdin input which we should remove first.
  fe.Inputs.clear(); 
  if (load_source(fe, temp_dir, program)!=0)
    return CL_OUT_OF_HOST_MEMORY;

  CodeGenOptions &cg = pocl_build.getCodeGenOpts();
  cg.EmitOpenCLArgMetadata = true;
  cg.StackRealignment = true;

  // TODO: use pch: it is possible to disable the strict checking for
  // the compilation flags used to compile it and the current translation
  // unit via the preprocessor options directly.

  bool success = true;
  clang::CodeGenAction *action = NULL;
  action = new clang::EmitLLVMOnlyAction(GlobalContext());
  success |= CI.ExecuteAction(*action);

  SourceManager &source_manager = CI.getSourceManager();
  for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->err_begin(),
       e = diagsBuffer->err_end(); i != e; ++i)
    {
      ss_build_log << "error: " << (*i).first.printToString(source_manager)
                   << ": " << (*i).second << std::endl;
    }
  for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->warn_begin(),
       e = diagsBuffer->warn_end(); i != e; ++i)
    {
      ss_build_log << "warning: " << (*i).first.printToString(source_manager)
                   << ": " << (*i).second << std::endl;
    }
  pocl_create_or_append_file(build_log_filename.str().c_str(),
                                ss_build_log.str().c_str());
  std::cerr << ss_build_log.str();

  // FIXME: memleak, see FIXME below
  if (!success) return CL_BUILD_PROGRAM_FAILURE;

  llvm::Module **mod = (llvm::Module **)&program->llvm_irs[device_i];
  if (*mod != NULL)
    delete (llvm::Module*)*mod;

#if LLVM_VERSION_MAJOR==3 && LLVM_VERSION_MINOR<6
  *mod = action->takeModule();
#else
  *mod = action->takeModule().release();
#endif

  if (*mod == NULL)
    return CL_BUILD_PROGRAM_FAILURE;

  /* Always retain program.bc. Its required in clBuildProgram */
  write_temporary_file(*mod, binary_file_name);

  // FIXME: cannot delete action as it contains something the llvm::Module
  // refers to. We should create it globally, at compiler initialization time.
  //delete action;

  return CL_SUCCESS;
}

int pocl_llvm_get_kernel_arg_metadata(const char* kernel_name,
                                      llvm::Module *input,
                                      cl_kernel kernel)
{

  // find the right kernel in "opencl.kernels" metadata
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  llvm::MDNode *kernel_metadata = NULL;

  // Not sure what to do in this case
  if (!opencl_kernels) return -1;

  for (unsigned i = 0, e = opencl_kernels->getNumOperands(); i != e; ++i) {
    llvm::MDNode *kernel_iter = opencl_kernels->getOperand(i);

    llvm::Function *kernel_prototype = llvm::cast<llvm::Function>(kernel_iter->getOperand(0));
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
      llvm::Value *meta_arg_value = meta_node->getOperand(j);
      struct pocl_argument_info* current_arg = &kernel->arg_info[j-1];

      if (isa<ConstantInt>(meta_arg_value) && meta_name=="kernel_arg_addr_space") {
        assert(has_meta_for_every_arg && "kernel_arg_addr_space meta incomplete");
        kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
        //std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
        llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
        uint64_t val = m->getLimitedValue(UINT_MAX);
        //std::cout << "with value: " << val << std::endl;
        if(bitcode_is_spir) {
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
          }
        }
      }
      else if (isa<MDString>(meta_arg_value)) {
        //std::cout << "is MDString" << std::endl;
        llvm::MDString *m = llvm::cast<MDString>(meta_arg_value);
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
                                  const char* device_tmpdir, 
                                  char* descriptor_filename,
                                  int * errcode)
{

  int i;
  llvm::Module *input = NULL;
  char tmpdir[POCL_FILENAME_LENGTH];

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

  (void) snprintf(descriptor_filename, POCL_FILENAME_LENGTH,
                    "%s/%s/descriptor.so", device_tmpdir, kernel_name);

  snprintf(tmpdir, POCL_FILENAME_LENGTH, "%s/%s",
            device_tmpdir, kernel_name);

  if (access(tmpdir, F_OK) != 0)
    mkdir(tmpdir, S_IRWXU);

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
#if (defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4)
  const std::string &ModuleDataLayout = input->getDataLayout();
#else
  const std::string &ModuleDataLayout = input->getDataLayout()->getStringRepresentation();
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
        locals.push_back(i);
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
      if (KernelSizeInfo->getOperand(0) == kernel_function) {
        reqdx = (llvm::cast<ConstantInt>
                 (KernelSizeInfo->getOperand(1)))->getLimitedValue();
        reqdy = (llvm::cast<ConstantInt>
                 (KernelSizeInfo->getOperand(2)))->getLimitedValue();
        reqdz = (llvm::cast<ConstantInt>
                 (KernelSizeInfo->getOperand(3)))->getLimitedValue();
      }
    }
  }
  kernel->reqd_wg_size[0] = reqdx;
  kernel->reqd_wg_size[1] = reqdy;
  kernel->reqd_wg_size[2] = reqdz;
  
#ifndef ANDROID
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous devices which need
  // these definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now
  std::string kobj_s = descriptor_filename; 
  kobj_s += ".kernel_obj.c";

  if(access(kobj_s.c_str(), F_OK) != 0)
    {
      FILE *kobj_c = fopen( kobj_s.c_str(), "wc");

      fprintf(kobj_c, "\n #include <pocl_device.h>\n");

      fprintf(kobj_c,
        "void _%s_workgroup(void** args, struct pocl_context*);\n", kernel_name);
      fprintf(kobj_c,
        "void _%s_workgroup_fast(void** args, struct pocl_context*);\n", kernel_name);

      fprintf(kobj_c,
        "__attribute__((address_space(3))) __kernel_metadata _%s_md = {\n", kernel_name);
      fprintf(kobj_c,
        "     \"%s\", /* name */ \n", kernel_name );
      fprintf(kobj_c,"     %d, /* num_args */\n", kernel->num_args);
      fprintf(kobj_c,"     %d, /* num_locals */\n", kernel->num_locals);
#if 0
      // These are not used anymore. The launcher knows the arguments
      // and sets them up, the device just obeys and launches with
      // whatever arguments it gets. Remove if none of the private
      // branches need them neither.
      fprintf( kobj_c," #if _%s_NUM_LOCALS != 0\n",   kernel_name  );
      fprintf( kobj_c,"     _%s_LOCAL_SIZE,\n",       kernel_name  );
      fprintf( kobj_c," #else\n"    );
      fprintf( kobj_c,"     {0}, \n"    );
      fprintf( kobj_c," #endif\n"    );
      fprintf( kobj_c,"     _%s_ARG_IS_LOCAL,\n",    kernel_name  );
      fprintf( kobj_c,"     _%s_ARG_IS_POINTER,\n",  kernel_name  );
      fprintf( kobj_c,"     _%s_ARG_IS_IMAGE,\n",    kernel_name  );
      fprintf( kobj_c,"     _%s_ARG_IS_SAMPLER,\n",  kernel_name  );
#endif
      fprintf( kobj_c,"     _%s_workgroup_fast\n",   kernel_name  );
      fprintf( kobj_c," };\n");
      fclose(kobj_c);
   }
#endif

  pocl_llvm_get_kernel_arg_metadata(kernel_name, input, kernel);

  return 0;
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

  return TheTarget->createTargetMachine(TheTriple.getTriple(),
                                        MCPU, FeaturesStr, GetTargetOptions(),
                                        Reloc::PIC_, CodeModel::Default,
                                        CodeGenOpt::Aggressive);
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
(cl_device_id device, std::string module_data_layout)
{
  static std::map<cl_device_id, PassManager*> kernel_compiler_passes;

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
    initializeIPA(Registry);
    initializeTransformUtils(Registry);
    initializeInstCombine(Registry);
    initializeInstrumentation(Registry);
    initializeTarget(Registry);
  }

#if !(defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4)
  // Scalarizer is in LLVM upstream since 3.4.
  const bool SCALARIZE = pocl_is_option_set("POCL_SCALARIZE_KERNELS");
#else
  const bool SCALARIZE = false;
#endif

#ifndef LLVM_3_2
  StringMap<llvm::cl::Option*> opts;
  llvm::cl::getRegisteredOptions(opts);
#endif

  PassManager *Passes = new PassManager();

  // Need to setup the target info for target specific passes. */
  TargetMachine *Machine = GetTargetMachine(device);
  // Add internal analysis passes from the target machine.
#ifndef LLVM_3_2
  if (Machine != NULL)
    Machine->addAnalysisPasses(*Passes);
#endif

  if (module_data_layout != "") {
#if (defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4)
    Passes->add(new DataLayout(module_data_layout));
#elif (defined LLVM_3_5)
    Passes->add(new DataLayoutPass(DataLayout(module_data_layout)));
#else
    Passes->add(new DataLayoutPass());
#endif
  }

  /* Disables automated generation of libcalls from code patterns. 
     TCE doesn't have a runtime linker which could link the libs later on.
     Also the libcalls might be harmful for WG autovectorization where we 
     want to try to vectorize the code it converts to e.g. a memset or 
     a memcpy */
  TargetLibraryInfo *TLI = new TargetLibraryInfo(triple);
  TLI->disableAllFunctions();
  Passes->add(TLI);

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
  passes.push_back("automatic-locals");
  passes.push_back("flatten");
  passes.push_back("always-inline");
  passes.push_back("globaldce");
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
  passes.push_back("allocastoentry");
  passes.push_back("workgroup");
  passes.push_back("target-address-spaces");
  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running the
  // standard LLVM optimizations.
  passes.push_back("simplifycfg");
  //passes.push_back("print-module");

  /* This is a beginning of the handling of the fine-tuning parameters.
   * TODO: POCL_KERNEL_COMPILER_OPT_SWITCH
   * TODO: POCL_VECTORIZE_WORK_GROUPS
   * TODO: POCL_VECTORIZE_VECTOR_WIDTH
   * TODO: POCl_VECTORIZE_NO_FP
   */
  const std::string wg_method = 
    pocl_get_string_option("POCL_WORK_GROUP_METHOD", "auto");

#ifndef LLVM_3_2
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
        }
      passes.push_back("mem2reg");
      passes.push_back("loop-vectorize");
      passes.push_back("slp-vectorizer");
    } 
#endif

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

#if defined(LLVM_3_2) || defined(LLVM_3_3)
          // SimplifyLibCalls has been removed in LLVM 3.4.
          Builder.DisableSimplifyLibCalls = true;
#endif
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

/* This is used to communicate the work-group dimensions command-line parameter to the 
   workitem loop. */
namespace pocl {
extern llvm::cl::list<int> LocalSize;
} 

/**
 * Return the OpenCL C built-in function library bitcode
 * for the given device.
 */
static llvm::Module*
kernel_library
(cl_device_id device, llvm::Module* root)
{
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  static std::map<cl_device_id, llvm::Module*> libs;

  Triple triple(device->llvm_target_triplet);

  if (libs.find(device) != libs.end())
    {
      return libs[device];
    }

  // TODO sync with Nat Ferrus' indexed linking
  std::string kernellib;
  if (pocl_get_bool_option("POCL_BUILDING", 0))
    {
      kernellib = BUILDDIR;
      kernellib += "/lib/kernel/";
      // TODO: get this from the target triplet: TCE, cellspu
      if (triple.getArch() == Triple::tce) 
        {
          kernellib += "tce";
        }
#ifdef LLVM_3_2 
      else if (triple.getArch() == Triple::cellspu) 
        {
          kernellib += "cellspu";
        }
#endif
      else 
        {
          kernellib += "host";
        }
      kernellib += "/kernel-"; 
      kernellib += device->llvm_target_triplet;
      kernellib +=".bc";   
    }
  else
    {
      kernellib = PKGDATADIR;
      kernellib += "/kernel-";
      kernellib += device->llvm_target_triplet;
      kernellib += ".bc";
    }

  SMDiagnostic Err;
  llvm::Module *lib = ParseIRFile(kernellib.c_str(), Err, *GlobalContext());
  assert (lib != NULL);
  libs[device] = lib;

  return lib;
}

/* This is used to control the kernel we want to process in the kernel compilation. */
extern cl::opt<std::string> KernelName;

int pocl_llvm_generate_workgroup_function(cl_device_id device,
                                          cl_kernel kernel,
                                          size_t local_x, size_t local_y, size_t local_z,
                                          const char* parallel_filename,
                                          const char* kernel_filename)
{
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

#ifdef DEBUG_POCL_LLVM_API        
  printf("### calling the kernel compiler for kernel %s local_x %zu "
         "local_y %zu local_z %zu parallel_filename: %s\n",
         kernel->name, local_x, local_y, local_z, parallel_filename);
#endif

  Triple triple(device->llvm_target_triplet);

  SMDiagnostic Err;
  std::string errmsg;

  // Link the kernel and runtime library
  llvm::Module *input = NULL;
  if (kernel->program->llvm_irs != NULL && 
      kernel->program->llvm_irs[device->dev_id] != NULL) 
    {
#ifdef DEBUG_POCL_LLVM_API        
      printf("### cloning the preloaded LLVM IR\n");
#endif
      input = 
        llvm::CloneModule
        ((llvm::Module*)kernel->program->llvm_irs[device->dev_id]);
    }
  else
    {
#ifdef DEBUG_POCL_LLVM_API        
      printf("### loading the kernel bitcode from disk\n");
#endif
      input = ParseIRFile(kernel_filename, Err, *GlobalContext());
    }

  // Later this should be replaced with indexed linking of source code
  // and/or bitcode for each kernel.
  llvm::Module *libmodule = kernel_library(device, input);
  assert (libmodule != NULL);
  link(input, libmodule);

  /* Now finally run the set of passes assembled above */
  // TODO pass these as parameters instead, this is not thread safe!
  pocl::LocalSize.clear();
  pocl::LocalSize.addValue(local_x);
  pocl::LocalSize.addValue(local_y);
  pocl::LocalSize.addValue(local_z);
  KernelName = kernel->name;

#if (defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4)
  kernel_compiler_passes(device, input->getDataLayout()).run(*input);
#else
  kernel_compiler_passes(device,
                         input->getDataLayout()->getStringRepresentation())
                        .run(*input);
#endif

  // TODO: don't write this once LLC is called via API, not system()
  write_temporary_file(input, parallel_filename);

#ifndef LLVM_3_2
  // In LLVM 3.2 the Linker object deletes the associated Modules.
  // If we delete here, it will crash.
  /* OPTIMIZE: store the fully linked work-group function llvm::Module 
     and pass it to code generation without writing to disk. */
#endif

  return 0;
}

void
pocl_update_program_llvm_irs(cl_program program,
                             cl_device_id device, const char* program_filename)
{
  SMDiagnostic Err;

  program->llvm_irs[device->dev_id] =
              ParseIRFile(program_filename, Err, *GlobalContext());
}

void pocl_llvm_update_binaries (cl_program program) {

  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  // Dump the LLVM IR Modules to memory buffers. 
  assert (program->llvm_irs != NULL);
#ifdef DEBUG_POCL_LLVM_API        
  printf("### refreshing the binaries of the program %p\n", program);
#endif

   for (size_t i = 0; i < program->num_devices; ++i)
    {
      assert (program->llvm_irs[i] != NULL);

      std::string binary_filename =
        std::string(program->temp_dir) + "/" + 
        program->devices[i]->short_name + "/" +
        POCL_PROGRAM_BC_FILENAME;

      write_temporary_file((llvm::Module*)program->llvm_irs[i],
                           binary_filename.c_str()); 

      FILE *binary_file = fopen(binary_filename.c_str(), "r");
      if (binary_file == NULL)        
        POCL_ABORT("Failed opening the binary file.");

      fseek(binary_file, 0, SEEK_END);
      
      program->binary_sizes[i] = ftell(binary_file);
      fseek(binary_file, 0, SEEK_SET);

      unsigned char *binary = (unsigned char *) malloc(program->binary_sizes[i]);
      if (binary == NULL)
        POCL_ABORT("Failed allocating memory for the binary.");

      size_t n = fread(binary, 1, program->binary_sizes[i], binary_file);
      if (n < program->binary_sizes[i])
        POCL_ABORT("Failed reading the binary from disk to memory.");
      program->binaries[i] = binary;

      fclose (binary_file);

#ifdef DEBUG_POCL_LLVM_API        
      printf("### binary for device %zi was of size %zu\n", i, program->binary_sizes[i]);
#endif

    }
}

int
pocl_llvm_get_kernel_names( cl_program program, const char **knames, unsigned max_num_krn )
{
  llvm::MutexGuard lockHolder(kernelCompilerLock);
  InitializeLLVM();

  // TODO: is it safe to assume every device (i.e. the index 0 here)
  // has the same set of programs & kernels?
  llvm::Module *mod = (llvm::Module *) program->llvm_irs[0];
  llvm::NamedMDNode *md = mod->getNamedMetadata("opencl.kernels");
  assert(md);

  unsigned i;
  for (i=0; i<md->getNumOperands(); i++) {
    assert( md->getOperand(i)->getOperand(0) != NULL);
    llvm::Function *k = cast<Function>(md->getOperand(i)->getOperand(0));
    if (i<max_num_krn)
      knames[i]= k->getName().data();
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
    SMDiagnostic Err;
#if defined LLVM_3_2 || defined LLVM_3_3
    std::string error;
    tool_output_file outfile(outfilename, error, 0);
#elif defined LLVM_3_4 || defined LLVM_3_5
    std::string error;
    tool_output_file outfile(outfilename, error, F_Binary);
#else
    std::error_code error;
    tool_output_file outfile(outfilename, error, F_Binary);
#endif
    llvm::Triple triple(device->llvm_target_triplet);
    llvm::TargetMachine *target = GetTargetMachine(device);
    llvm::Module *input = ParseIRFile(infilename, Err, *GlobalContext());

    llvm::PassManager PM;
    llvm::TargetLibraryInfo *TLI = new TargetLibraryInfo(triple);
    PM.add(TLI);
    if (target != NULL) {
#if defined LLVM_3_2
      PM.add(new TargetTransformInfo(target->getScalarTargetTransformInfo(),
                                     target->getVectorTargetTransformInfo()));
#else
      target->addAnalysisPasses(PM);
#endif
    }

    // TODO: get DataLayout from the 'device'
#if defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4
    const DataLayout *TD = NULL;
    if (target != NULL)
      TD = target->getDataLayout();
    if (TD != NULL)
        PM.add(new DataLayout(*TD));
    else
        PM.add(new DataLayout(input));
#endif
    // TODO: better error check
    formatted_raw_ostream FOS(outfile.os());
    llvm::MCContext *mcc;
    if(target->addPassesToEmitMC(PM, mcc, FOS, llvm::TargetMachine::CGFT_ObjectFile))
        return 1;

    PM.run(*input);
    outfile.keep();

    return 0;
}
/* vim: set ts=4 expandtab: */

