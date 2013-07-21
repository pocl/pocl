#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <sys/stat.h>

// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros, 
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm.h"

using namespace clang;
using namespace llvm;

/* "emulate" the pocl_build script.
 * This compiles an .cl file into LLVM IR 
 * (the "program.bc") file.
 * unlike the script, a intermediate preprocessed 
 * program.bc.i file is not produced.
 */
int call_pocl_build( cl_device_id device, 
                     const char* source_file_name,
                     const char* binary_file_name,
                     const char* device_tmpdir,
                     const char* user_options )

{
   
  CompilerInstance CI;
  CompilerInvocation &pocl_build = CI.getInvocation();

  //TODO: why does getLangOpts return a pointer, when the other getXXXOpts() return a reference?
  LangOptions *la = pocl_build.getLangOpts();
  // the per-file types don't seem to override this :/
  // FIXME: setting of the language standard (OCL 1.2, etc.) left as 'undefined' here
  la->FakeAddressSpaceMap=true;
  pocl_build.setLangDefaults(*la, clang::IK_OpenCL);
  
  // FIXME: print out any diagnostics to stdout for now. These should go to a buffer for the user
  // to dig out. (and probably to stdout too, overridable with environment variables) 
  #ifdef LLVM_3_2
  CI.createDiagnostics(0, NULL);
  #else
  CI.createDiagnostics();
  #endif 
 
  FrontendOptions &fe = pocl_build.getFrontendOpts();
  fe.Inputs.push_back(FrontendInputFile(source_file_name, clang::IK_OpenCL));
  fe.OutputFile=std::string(binary_file_name);

  PreprocessorOptions &pp = pocl_build.getPreprocessorOpts();
  // FIXME: these paths are wrong!
  pp.Includes.push_back(BUILDDIR "/include/x86_64/types.h");
  pp.Includes.push_back(BUILDDIR "/../pocl/include/_kernel.h");

  clang::TargetOptions &ta = pocl_build.getTargetOpts();
  // FIXME:!
  ta.Triple = llvm::sys::getDefaultTargetTriple();
  
  CodeGenOptions &cg = pocl_build.getCodeGenOpts();
  // This is the "-O" flag for clang - setting it to 3 breaks barriers,
  // leaving it low causes slow code.
  cg.OptimizationLevel = 2;

  // TODO: switch to EmitLLVMOnlyAction, when intermediate file is not needed
  CodeGenAction *action = new clang::EmitBCAction(&llvm::getGlobalContext());
  return CI.ExecuteAction(*action) ? CL_SUCCESS:CL_BUILD_PROGRAM_FAILURE;
}

/* Emulate calling the pocl_kernel script.
 * 
 * This is documented as:  
# pocl-kernel - Examine a OpenCL bytecode and generate a loadable module
#               with kernel function information.
 * 
 */
int call_pocl_kernel(cl_program program, 
                     cl_kernel kernel,
                     int device_i,     
                     const char* kernel_name,
                     const char* device_tmpdir, 
                     char* descriptor_filename,
                     int *errcode )
{

  int error,n, i;
  llvm::Module *input;
  LLVMContext &Context = getGlobalContext();
  SMDiagnostic Err;
  FILE *binary_file;
  char binary_filename[POCL_FILENAME_LENGTH];
  char object_filename[POCL_FILENAME_LENGTH];
  char tmpdir[POCL_FILENAME_LENGTH];

  const char *triple;

  //TODO: a device should *allways* know its triple.
  if( program->devices[device_i]->llvm_target_triplet != NULL )
    triple = program->devices[device_i]->llvm_target_triplet;
  else
    triple = sys::getDefaultTargetTriple().c_str();

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

  n = fwrite(program->binaries[device_i], 1,
             program->binary_sizes[device_i], binary_file);
  if (n < program->binary_sizes[device_i])
    return (CL_OUT_OF_HOST_MEMORY);
  
  fclose(binary_file); 
  input = ParseIRFile(binary_filename, Err, Context);

  PassManager Passes;
  DataLayout *TD = 0;
  const std::string &ModuleDataLayout = input->getDataLayout();
  if (!ModuleDataLayout.empty())
    TD = new DataLayout(ModuleDataLayout);

  llvm::Function *kernel_function = input->getFunction(kernel_name);
  assert( kernel_function && "TODO: make better check here");

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
    if (i->getName().startswith(funcName + ".")) {
      // Additional checks might be needed here. For now
      // we assume any global starting with kernel name
      // is declaring a local variable.
      locals.push_back(i);
    }
  }

  kernel->num_locals=locals.size();

  /* This is from clCreateKernel.c */
  /* Temporary store for the arguments that are set with clSetKernelArg. */
  kernel->dyn_arguments =
    (struct pocl_argument *) malloc ((kernel->num_args + kernel->num_locals) *
                                     sizeof (struct pocl_argument));
  /* Initialize kernel "dynamic" arguments (in case the user doesn't). */
  for (int i = 0; i < kernel->num_args; ++i)
    {
      kernel->dyn_arguments[i].value = NULL;
      kernel->dyn_arguments[i].size = 0;
    }

  /* Fill up automatic local arguments. */
  for (int i = 0; i < kernel->num_locals; ++i)
    {
      kernel->dyn_arguments[kernel->num_args + i].value = NULL;
      kernel->dyn_arguments[kernel->num_args + i].size =
        TD->getTypeAllocSize(locals[i]->getInitializer()->getType());
    }

  // TODO: if the scripts are dumped, consider just having one list of 
  // enumerations, declaring what sort the arguments are
  kernel->arg_is_pointer = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  kernel->arg_is_local = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  kernel->arg_is_image = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  kernel->arg_is_sampler = (cl_int*)malloc( sizeof(cl_int)*kernel->num_args );
  
  // This is from GenerateHeader.cc
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
          p->getAddressSpace() == POCL_ADDRESS_SPACE_CONSTANT)
        kernel->arg_is_local[i] = false;
      else
        kernel->arg_is_local[i] = true;
    } else {
      kernel->arg_is_pointer[i] = false;
      kernel->arg_is_local[i] = false;
    }
    
    if (t->isPointerTy()) {
      if (t->getPointerElementType()->isStructTy()) {
        std::string name = t->getPointerElementType()->getStructName().str();
        if (name == "opencl.image2d_t" || name == "opencl.image3d_t" || 
            name == "opencl.image1d_t" || name == "struct.dev_image_t") {
          kernel->arg_is_image[i] = true;
          kernel->arg_is_pointer[i] = false;
          kernel->arg_is_local[i] = false;
        }
        if (name == "opencl.sampler_t_") {
          kernel->arg_is_sampler[i] = true;
          kernel->arg_is_pointer[i] = false;
          kernel->arg_is_local[i] = false;
        }
      }
    }
    i++;  
  }
  
  // TODO: fill 'kernel->reqd_wg_size'!

  // Generate the kernel_obj.c file
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now
  std::string kobj_s = descriptor_filename; 
  kobj_s += ".kernel_obj.c"; 
  FILE *kobj_c = fopen( kobj_s.c_str(), "wc");
 
  fprintf( kobj_c, "\n #include <pocl_device.h>\n");

  fprintf( kobj_c,
    "void _%s_workgroup(void** args, struct pocl_context*);\n", kernel_name );
  fprintf( kobj_c,
    "void _%s_workgroup_fast(void** args, struct pocl_context*);\n", kernel_name );

  fprintf( kobj_c,
    "__attribute__((address_space(3))) __kernel_metadata _%s_md = {\n", kernel_name );
  fprintf( kobj_c,
    "     \"%s\", /* name */ \n", kernel_name );
  fprintf( kobj_c,"     _%s_NUM_ARGS, /* num_args */\n",      kernel_name );
  fprintf( kobj_c,"     _%s_NUM_LOCALS, /* num_locals */\n",  kernel_name );
  fprintf( kobj_c," #if _%s_NUM_LOCALS != 0\n",   kernel_name  );
  fprintf( kobj_c,"     _%s_LOCAL_SIZE,\n",       kernel_name  );
  fprintf( kobj_c," #else\n"    );
  fprintf( kobj_c,"     {0}, \n"    );
  fprintf( kobj_c," #endif\n"    );
  fprintf( kobj_c,"     _%s_ARG_IS_LOCAL,\n",    kernel_name  );
  fprintf( kobj_c,"     _%s_ARG_IS_POINTER,\n",  kernel_name  );
  fprintf( kobj_c,"     _%s_ARG_IS_IMAGE,\n",    kernel_name  );
  fprintf( kobj_c,"     _%s_ARG_IS_SAMPLER,\n",  kernel_name  );
  fprintf( kobj_c,"     _%s_workgroup_fast\n",   kernel_name  );
  fprintf( kobj_c," };\n");
  fclose(kobj_c);
  
  return 0;
  
}

