#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Path.h"

// Note - LLVM/Clang uses symbols defined in Khronos' headers in macros, 
// causing compilation error if they are included before the LLVM headers.
#include "pocl_llvm.h"

using namespace clang;

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

  TargetOptions &ta = pocl_build.getTargetOpts();
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

