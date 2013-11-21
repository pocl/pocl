/* pocl_llvm_api.cc: C wrappers for calling the LLVM/Clang C++ APIs

   Copyright (c) 2013 Kalle Raiskila and
                      Pekka Jääskeläinen
   
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
#include "llvm/Linker.h"
#include "llvm/PassManager.h"
#include "llvm/Bitcode/ReaderWriter.h"

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

#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <sys/stat.h>

#include <iostream>
#include <vector>
#include <sstream>
#include <string>

// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros, 
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm.h"
#include "pocl_runtime_config.h"
#include "install-paths.h"
#include "LLVMUtils.h"

using namespace clang;
using namespace llvm;

#if defined LLVM_3_2 || defined LLVM_3_3
#include "llvm/Support/raw_ostream.h"
#define F_Binary llvm::raw_fd_ostream::F_Binary
#else
using llvm::sys::fs::F_Binary;
#endif

//#define DEBUG_POCL_LLVM_API

/* "emulate" the pocl_build script.
 * This compiles an .cl file into LLVM IR 
 * (the "program.bc") file.
 * unlike the script, a intermediate preprocessed 
 * program.bc.i file is not produced.
 */
int call_pocl_build(cl_device_id device, 
                    const char* source_file_name,
                    const char* binary_file_name,
                    const char* device_tmpdir,
                    const char* user_options)

{ 

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

  // TODO: call device->prepare_build() that can return
  // device-specific switches too, to replace the build_program()
  // API TCE uses to include the custom op macros.
  std::stringstream ss;

  if (device->init_build != NULL) 
    {
      assert (device_tmpdir != NULL);
      char *device_switches = 
        device->init_build (device->data, device_tmpdir);
      if (device_switches != NULL) 
        {
          ss << device_switches << " ";
        }
      free (device_switches);
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
      itemcstrs.push_back(itemstrs.back().c_str());
      ++i;
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
          // TODO: transfer the errors to clGetProgramBuildInfo
          std::cerr << "error: " << (*i).second << std::endl;
        }
      for (TextDiagnosticBuffer::const_iterator i = diagsBuffer->warn_begin(), 
             e = diagsBuffer->warn_end(); i != e; ++i) 
        {
          // TODO: transfer the warnings to clGetProgramBuildInfo
          std::cerr << "warning: " << (*i).second << std::endl;
        }
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

  // -Wno-format
  PreprocessorOptions &po = pocl_build.getPreprocessorOpts();
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
  CI.createDiagnostics(0, NULL);
#else
  CI.createDiagnostics();
#endif 
 
  FrontendOptions &fe = pocl_build.getFrontendOpts();
  // The CreateFromArgs created an stdin input which we should remove first.
  fe.Inputs.clear(); 
  fe.Inputs.push_back
    (FrontendInputFile(source_file_name, clang::IK_OpenCL));
  fe.OutputFile = std::string(binary_file_name);

  CodeGenOptions &cg = pocl_build.getCodeGenOpts();
  cg.EmitOpenCLArgMetadata = 1;

  // TODO: use pch: it is possible to disable the strict checking for
  // the compilation flags used to compile it and the current translation
  // unit via the preprocessor options directly.

  // TODO: switch to EmitLLVMOnlyAction, when intermediate file is not needed
  // Do not give the global context to EmitBCAction as that leads to the
  // image types clashing and a running number appended to them whenever a
  // new module with the opaque type is reloaded.
  CodeGenAction *action = new clang::EmitBCAction();
  bool success = CI.ExecuteAction(*action);
  return success ? CL_SUCCESS : CL_BUILD_PROGRAM_FAILURE;
}

/* Retrieve metadata of the given kernel in the program to populate the
 * cl_kernel object.
 */
int call_pocl_kernel(cl_program program, 
                     cl_kernel kernel,
                     int device_i,     
                     const char* kernel_name,
                     const char* device_tmpdir, 
                     char* descriptor_filename,
                     int */*errcode*/)
{

  int error, i;
  unsigned n;
  llvm::Module *input;
  SMDiagnostic Err;
  FILE *binary_file;
  char binary_filename[POCL_FILENAME_LENGTH];
  char tmpdir[POCL_FILENAME_LENGTH];

  assert(program->devices[device_i]->llvm_target_triplet && 
         "Device has no target triple set"); 

  snprintf (tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
            device_tmpdir, kernel_name);
  mkdir (tmpdir, S_IRWXU);

  error = snprintf(binary_filename, POCL_FILENAME_LENGTH,
                   "%s/kernel.bc",
                   tmpdir);
  error |= snprintf(descriptor_filename, POCL_FILENAME_LENGTH,
                   "%s/%s/descriptor.so", device_tmpdir, kernel_name);

  binary_file = fopen(binary_filename, "w+");
  if (binary_file == NULL)
    return (CL_OUT_OF_HOST_MEMORY);

  // TODO: dump the .bc only if we are using the kernel compiler
  // cache (POCL_LEAVE_TEMP_DIRS). Otherwise store a Module*
  // at clBuildProgram and reuse it here.
  n = fwrite(program->binaries[device_i], 1,
             program->binary_sizes[device_i], binary_file);
  if (n < program->binary_sizes[device_i])
    return (CL_OUT_OF_HOST_MEMORY);
  fclose(binary_file); 

  // Create own temporary context so we do not get duplicate image and sampler
  // opaque types to the context. They will get running numbers and
  // the name does not match anymore. 
  LLVMContext context;
  input = ParseIRFile(binary_filename, Err, context);
  if(!input) 
    {
      // TODO:
      raw_os_ostream os(std::cout);
      Err.print("pocl error: bad kernel file ", os);
      os.flush();
      exit(1);
    }
  
  PassManager Passes;
  DataLayout *TD = 0;
  const std::string &ModuleDataLayout = input->getDataLayout();
  if (!ModuleDataLayout.empty())
    TD = new DataLayout(ModuleDataLayout);

  llvm::Function *kernel_function = input->getFunction(kernel_name);
  assert(kernel_function && "TODO: make better check here");

  const llvm::Function::ArgumentListType &arglist = 
      kernel_function->getArgumentList();
  kernel->num_args=arglist.size();

  // This is from GenerateHeader.cc
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

  /* This is from clCreateKernel.c */
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

  // TODO: if the scripts are dumped, consider just having one list of 
  // enumerations, declaring what sort the arguments are
  kernel->arg_is_pointer = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  kernel->arg_is_local = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  kernel->arg_is_image = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  kernel->arg_is_sampler = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  
  // This is from GenerateHeader.cc
  i=0;
  for( llvm::Function::const_arg_iterator ii = arglist.begin(), 
                                          ee = arglist.end(); 
       ii != ee ; ii++)
  {
    Type *t = ii->getType();
  
    kernel->arg_is_image[i] = false;
    kernel->arg_is_sampler[i] = false;
 
    const PointerType *p = dyn_cast<PointerType>(t);
    if (p && !ii->hasByValAttr()) {
      kernel->arg_is_pointer[i] = true;
      // index 0 is for function attributes, parameters start at 1.
      if (p->getAddressSpace() == POCL_ADDRESS_SPACE_GLOBAL ||
          p->getAddressSpace() == POCL_ADDRESS_SPACE_CONSTANT ||
          pocl::is_image_type(*t) || pocl::is_sampler_type(*t))
        {
          kernel->arg_is_local[i] = false;
        }
      else
        {
          if (p->getAddressSpace() != POCL_ADDRESS_SPACE_LOCAL)
            {
              p->dump();
              assert(p->getAddressSpace() == POCL_ADDRESS_SPACE_LOCAL);
            }
          kernel->arg_is_local[i] = true;
        }
    } else {
      kernel->arg_is_pointer[i] = false;
      kernel->arg_is_local[i] = false;
    }

    if (pocl::is_image_type(*t))
      {
        kernel->arg_is_image[i] = true;
        kernel->arg_is_pointer[i] = false;
      } 
    else if (pocl::is_sampler_type(*t)) 
      {
        kernel->arg_is_sampler[i] = true;
        kernel->arg_is_pointer[i] = false;
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
  
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous devices which need
  // these definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now
  std::string kobj_s = descriptor_filename; 
  kobj_s += ".kernel_obj.c"; 
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
  
  return 0;
  
}

/* helpers copied from LLVM opt START */

static llvm::TargetOptions GetTargetOptions() {
  llvm::TargetOptions Options;
  /* TODO: propagate these from clBuildProgram options. */
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
  Options.PositionIndependentExecutable = EnablePIE;
  Options.EnableSegmentedStacks = SegmentedStacks;
  Options.UseInitArray = UseInitArray;
  Options.SSPBufferSize = SSPBufferSize;
#endif
  return Options;
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine
(Triple TheTriple, 
 std::string MCPU,
 const std::vector<std::string>& MAttrs=std::vector<std::string>()) {

  std::string Error;
  const Target *TheTarget = 
    TargetRegistry::lookupTarget("" /*MArch*/, TheTriple,
                                 Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
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
                                        Reloc::Default, CodeModel::Default,
                                        CodeGenOpt::Aggressive);
}

/* helpers copied from LLVM opt END */

/* The kernel compiler passes are at the moment not thread safe,
   ensure only one thread is using it at the time with a mutex. */
static pocl_lock_t kernel_compiler_lock = POCL_LOCK_INITIALIZER;

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
#ifndef LLVM_3_2
  StringMap<llvm::cl::Option*> opts;
  llvm::cl::getRegisteredOptions(opts);
#endif
  PassRegistry &Registry = *PassRegistry::getPassRegistry();

  const bool first_initialization_call = kernel_compiler_passes.size() == 0;

  if (first_initialization_call) 
    {
      // We have not initialized any pass managers for any device yet.
      // Run the global LLVM pass initialization functions.
      InitializeAllTargets();
      InitializeAllTargetMCs();

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

#ifndef LLVM_3_2
      llvm::cl::Option *O = opts["add-wi-metadata"];
      O->addOccurrence(1, StringRef("add-wi-metadata"), StringRef(""), false); 
#endif

    }

  PassManager *Passes = new PassManager();

  // Need to setup the target info for target specific passes. */
  TargetMachine *Machine = 
    GetTargetMachine(triple, device->llvm_cpu ? device->llvm_cpu : "");
  // Add internal analysis passes from the target machine.
#ifndef LLVM_3_2
  Machine->addAnalysisPasses(*Passes);
#endif

  if (module_data_layout != "")
    Passes->add(new DataLayout(module_data_layout));
 

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
  passes.push_back("mem2reg");
  passes.push_back("domtree");
  passes.push_back("workitem-handler-chooser");
  passes.push_back("break-constgeps");
  passes.push_back("automatic-locals");
  passes.push_back("flatten");
  passes.push_back("always-inline");
  passes.push_back("globaldce");
  passes.push_back("simplifycfg");
  passes.push_back("loop-simplify");
  passes.push_back("phistoallocas");
  passes.push_back("isolate-regions");
  passes.push_back("uniformity");
  passes.push_back("implicit-loop-barriers");
  passes.push_back("implicit-cond-barriers");
  passes.push_back("loop-barriers");
  passes.push_back("barriertails");
  passes.push_back("barriers");
  passes.push_back("isolate-regions");
  passes.push_back("wi-aa");
  passes.push_back("workitemrepl");
  passes.push_back("workitemloops");
  passes.push_back("allocastoentry");
  passes.push_back("workgroup");
  passes.push_back("target-address-spaces");

  /* This is a beginning of the handling of the fine-tuning parameters.
   * TODO: POCL_KERNEL_COMPILER_OPT_SWITCH
   * TODO: POCL_VECTORIZE_WORK_GROUPS
   * TODO: POCL_VECTORIZE_VECTOR_WIDTH
   * TODO: POCl_VECTORIZE_NO_FP
   */
  const std::string wg_method = 
    pocl_get_string_option("POCL_WORK_GROUP_METHOD", "auto");

  const bool wi_vectorizer = 
    pocl_get_bool_option("POCL_VECTORIZE_WORK_GROUPS", 0);


#ifndef LLVM_3_2
  if (wg_method == "loopvec")
    {
      if (kernel_compiler_passes.size() == 0) {
        // Set the options only once. TODO: fix it so that each
        // device can reset their own options. Now one cannot compile
        // with different options to different devices at one run.
   
        llvm::cl::Option *O = opts["vectorizer-min-trip-count"];
        assert(O && "could not find LLVM option 'vectorizer-min-trip-count'");
        O->addOccurrence(1, StringRef("vectorizer-min-trip-count"), StringRef("2"), false); 

        O = opts["scalarize-load-store"];
        assert(O && "could not find LLVM option 'scalarize-load-store'");
        O->addOccurrence(1, StringRef("scalarize-load-store"), StringRef(""), false); 

        O = opts["enable-scalarizer"];
        assert(O && "could not find LLVM option 'enable-scalarizer'");
        O->addOccurrence(1, StringRef("enable-scalarizer"), StringRef(""), false); 

#ifdef DEBUG_POCL_LLVM_API        
        printf ("### autovectorizer enabled\n");

        O = opts["debug-only"];
        assert(O && "could not find LLVM option 'debug'");
        O->addOccurrence(1, StringRef("debug-only"), StringRef("loop-vectorize"), false); 

#endif
      }

      passes.push_back("scalarizer");
      passes.push_back("mem2reg");
      passes.push_back("loop-vectorize");
      passes.push_back("slp-vectorizer");
    } 
  else if (wi_vectorizer) 
    {
      /* The legacy repl based WI autovectorizer. Deprecated but 
         for still needed by some legacy TTA machines. */
      passes.push_back("STANDARD_OPTS");
      passes.push_back("wi-vectorize");
      llvm::cl::Option *O;
      if (pocl_is_option_set("POCL_VECTORIZE_VECTOR_WIDTH") && 
          first_initialization_call) 
        {
          /* The options cannot be unset, it seems, so we must set them 
             only once, globally. TODO: check further if there is some way to
             unset the options so we can control them per kernel compilation. */
          O = opts["wi-vectorize-vector-width"];
          assert(O && "could not find LLVM option 'wi-vectorize-vector-width'");
          O->addOccurrence(1, StringRef("wi-vectorize-vector-width"), 
                           pocl_get_string_option("POCL_VECTORIZE_VECTOR_WIDTH", "0"), false); 

        }

      if (pocl_get_bool_option("POCL_VECTORIZE_NO_FP", 0) && 
          first_initialization_call) 
        {
          O = opts["wi-vectorize-no-fp"];
          assert(O && "could not find LLVM option 'wi-vectorize-no-fp'");
          O->addOccurrence(1, StringRef("wi-vectorize-no-fp"), StringRef(""), false); 
        }

      if (pocl_get_bool_option("POCL_VECTORIZE_MEM_ONLY", 0) && 
          first_initialization_call) 
        {
          O = opts["wi-vectorize-mem-ops-only"];
          assert(O && "could not find LLVM option 'wi-vectorize-mem-ops-only'");
          O->addOccurrence(1, StringRef("wi-vectorize-mem-ops-only"), StringRef(""), false); 
        }

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
          Builder.DisableSimplifyLibCalls = true;
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

/* kludge - this is the kernel dimensions command-line parameter to the workitem loop */
namespace pocl {
extern llvm::cl::list<int> LocalSize;
} 

/* This function links the input kernel LLVM bitcode and the
 * OpenCL kernel runtime library into one LLVM module, then
 * runs pocl's kernel compiler passes on that module to produce 
 * a function that executes all work-items in a work-group.
 *
 * Output is a LLVM bitcode file. 
 *
 * TODO: rename these functions for something more descriptive.
 * TODO: this is not thread-safe, it changes the LLVM global options to
 * control the compilation. We should enforce only one compilations is done
 * at a time or control the options through thread safe methods.
 */
int call_pocl_workgroup(cl_device_id device,
                        char* function_name, 
                        size_t local_x, size_t local_y, size_t local_z,
                        const char* parallel_filename,
                        const char* kernel_filename)
{
  // TODO pass these as parameters instead, this is not thread safe!
  pocl::LocalSize.clear();
  pocl::LocalSize.addValue(local_x);
  pocl::LocalSize.addValue(local_y);
  pocl::LocalSize.addValue(local_z);

  Triple triple(device->llvm_target_triplet);

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
      // TODO: vefify this is the correct place!
      kernellib = PKGDATADIR;
      kernellib += KERNEL_DIR;
      kernellib += "/kernel-";
      kernellib += device->llvm_target_triplet;
      kernellib += ".bc";
    }

  // Have one LLVMContext per compilation to be (more) thread safe.
  LLVMContext Context;
  SMDiagnostic Err;
  std::string errmsg;

  // Link the kernel and runtime library
  llvm::Module *input = ParseIRFile(kernel_filename, Err, Context);
  llvm::Module *libmodule = ParseIRFile(kernellib, Err, Context);
  assert (libmodule != NULL);
#ifdef LLVM_3_2
  Linker TheLinker("pocl", input);
  TheLinker.LinkInModule(libmodule, &errmsg);
#else
  Linker TheLinker(input);
  TheLinker.linkInModule(libmodule, &errmsg);
#endif
  llvm::Module *linked_bc = TheLinker.getModule();

  assert (linked_bc != NULL);

  /* Now finally run the set of passes assembled above */
  std::string ErrorInfo;
  tool_output_file *Out = new tool_output_file(parallel_filename, 
                                               ErrorInfo, 
                                               F_Binary);

  POCL_LOCK(kernel_compiler_lock);
  kernel_compiler_passes(device, linked_bc->getDataLayout()).run(*linked_bc);
  POCL_UNLOCK(kernel_compiler_lock);

  WriteBitcodeToFile(linked_bc, Out->os()); 

  Out->keep();
  delete Out;
#ifndef LLVM_3_2
  // In LLVM 3.2 the Linker object deletes the associated Modules.
  // If we delete here, it will crash.
  delete linked_bc;
#endif

  return 0;
}
