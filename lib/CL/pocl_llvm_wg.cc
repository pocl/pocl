/* pocl_llvm_wg.cc: part of pocl LLVM API dealing with parallel.bc,
   optimization passes and codegen.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2019 Pekka Jääskeläinen
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/TargetParser/Triple.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/Support/Casting.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/IR/Verifier.h>
#include <llvm/PassInfo.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Linker/Linker.h>
#if LLVM_MAJOR >= 18
#include <llvm/Support/CodeGen.h>
#endif
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Utils/Cloning.h>
// legacy PassManager is needed for CodeGen, even if new PM is used for IR
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#if LLVM_MAJOR >= 18
#include <llvm/Frontend/Driver/CodeGenOptions.h>
#endif

#include "LLVMUtils.h"
POP_COMPILER_DIAGS

#include "common.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_llvm_api.h"
#include "pocl_spir.h"
#include "pocl_util.h"

#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "linker.h"
#include "spirv_parser.hh"

// Enable to get the LLVM pass execution timing report dumped to console after
// each work-group IR function generation. Requires LLVM > 7.
// #define DUMP_LLVM_PASS_TIMINGS

// Enable extra debugging of the new PM's run: prints each pass as it's run,
// plus the analysis it required, in a "tree like" fashion.
// #define DEBUG_NEW_PASS_MANAGER

// Use a separate PM instance to run the default optimization pipeline
// TODO: this MUST be left enabled for now; disabling causes a few tests fail
// should be investigated
#define SEPARATE_OPTIMIZATION_FROM_POCL_PASSES

// use a separate instance of llvm::TargetMachine; disabling this
// may cause test failures / random crashes / ASanitizer complaints
#define PER_STAGE_TARGET_MACHINE

using namespace llvm;

static bool verifyIR() {
  return pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT);
}

static bool enableDebugLogs() {
  bool Enable = pocl_get_bool_option("POCL_DEBUG_LLVM_PASSES", 0);
  Enable |= pocl_is_option_set("POCL_DEBUG_LLVM_OPTS");
  return Enable;
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine *GetTargetMachine(const char* TTriple,
                                       const char* MCPU = "",
                                       const char* Features = "") {

  std::string Error;

  const Target *TheTarget = TargetRegistry::lookupTarget(TTriple,
                                                         Error);

  // OpenASIP targets are not in the registry
  if (!TheTarget) {
    return nullptr;
  }

  TargetMachine *TM = TheTarget->createTargetMachine(
      TTriple, MCPU, Features, TargetOptions(),
      Reloc::PIC_, CodeModel::Small,
#if LLVM_MAJOR >= 18
      CodeGenOptLevel::Aggressive);
#else
      CodeGenOpt::Aggressive);
#endif

  assert(TM != NULL && "llvm target has no targetMachine constructor");

  return TM;
}


class PoCLModulePassManager {
  // Create the analysis managers.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  ModulePassManager PM;
  std::unique_ptr<TargetLibraryInfoImpl> TLII;
  std::unique_ptr<StandardInstrumentations> SI;
#ifdef PER_STAGE_TARGET_MACHINE
  std::unique_ptr<llvm::TargetMachine> Machine;
#endif
  PrintPassOptions PrintPassOpts;
  PassInstrumentationCallbacks PIC;
  llvm::LLVMContext Context; // for SI
  std::unique_ptr<PassBuilder> PassB;
  unsigned OptimizeLevel;
  unsigned SizeLevel;
  bool Vectorize = false;

public:
  PoCLModulePassManager() = default;
  llvm::Error build(std::string PoclPipeline, unsigned OLevel, unsigned SLevel,
                    bool EnableVectorizers,
#ifndef PER_STAGE_TARGET_MACHINE
                    TargetMachine *TM,
#endif
                    cl_device_id Dev);
  void run(llvm::Module &Bitcode);
};

llvm::Error PoCLModulePassManager::build(std::string PoclPipeline,
                                         unsigned OLevel, unsigned SLevel,
                                         bool EnableVectorizers,
#ifndef PER_STAGE_TARGET_MACHINE
                                         TargetMachine *TM,
#endif
                                         cl_device_id Dev) {

#ifdef PER_STAGE_TARGET_MACHINE
  Machine.reset(GetTargetMachine(Dev->llvm_target_triplet, Dev->llvm_cpu));
  TargetMachine *TM = Machine.get();
#endif

  PipelineTuningOptions PTO;
  // TODO: Does this affect the loop unroller of the vectorizer as well? We
  // might want to enable it in the default case.
  PTO.LoopUnrolling = false;
  PTO.UnifiedLTO = false;
  PTO.SLPVectorization = PTO.LoopVectorization = Vectorize = EnableVectorizers;
  OptimizeLevel = OLevel;
  SizeLevel = SLevel;

  PrintPassOpts.Verbose = false;
  PrintPassOpts.SkipAnalyses = true;
  PrintPassOpts.Indent = true;
  SI.reset(new StandardInstrumentations(Context,
                                        enableDebugLogs(), // debug logging
                                        verifyIR(),        // verify each
                                        PrintPassOpts));
  SI->registerCallbacks(PIC, &MAM);

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  PassB.reset(new PassBuilder(TM, PTO, std::nullopt, &PIC));
  PassBuilder &PB = *PassB.get();

#if 0
  // Add LibraryInfo.
  TLII.reset(llvm::driver::createTLII(TargetTriple, CodeGenOpts.getVecLib()));
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(*TLII));

  bool res;
  if (Machine) {
    PB.registerPipelineParsingCallback(
        [](::llvm::StringRef Name, ::llvm::FunctionPassManager &FPM,
           llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {
          if (Name == "require<targetir>") {
            FPM.addPass(RequireAnalysisPass<TargetIRAnalysis, llvm::Function>());
            return true;
          } else
            return false;
        });
    PB.registerAnalysisRegistrationCallback(
        [this](::llvm::FunctionAnalysisManager &FAM) {
          FAM.registerPass([=] { return TargetIRAnalysis(Machine->getTargetIRAnalysis()); });
        });

    res = FAM.registerPass([=] { return TargetIRAnalysis(Machine->getTargetIRAnalysis()); });
    assert(res && "TIRA already registered!");
  }

  // early register here, this will automatically override later registrations
  res = FAM.registerPass([=] { return TargetLibraryAnalysis(*TLII); });
  assert(res && "TLII already registered!");
  PB.registerAnalysisRegistrationCallback(
      [this](::llvm::FunctionAnalysisManager &FAM) {
        FAM.registerPass([=] { return TargetLibraryAnalysis(*TLII); });
      });

  PoclPipeline = "function(require<targetir>),function(require<targetlibinfo>)," + PoclPipeline;
#endif

  pocl::registerFunctionAnalyses(PB);
  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  pocl::registerPassBuilderPasses(PB);

#ifndef SEPARATE_OPTIMIZATION_FROM_POCL_PASSES
  OptimizationLevel Opt;
  if (SizeLevel > 0)
    PoclPipeline += ",default<Os>";
  else {
    switch (OptimizeLevel) {
    case 0:
      PoclPipeline += ",default<O0>";
      break;
    case 1:
      PoclPipeline += ",default<O1>";
      break;
    case 2:
      PoclPipeline += ",default<O2>";
      break;
    case 3:
    default:
      PoclPipeline += ",default<O3>";
      break;
    }
  }
#endif

  return PB.parsePassPipeline(PM, StringRef(PoclPipeline));
}

void PoCLModulePassManager::run(llvm::Module &Bitcode) {
  PM.run(Bitcode, MAM);
#ifdef SEPARATE_OPTIMIZATION_FROM_POCL_PASSES
  populateModulePM(nullptr, (void *)&Bitcode, OptimizeLevel, SizeLevel,
                   Vectorize, Machine.get());
#endif
}

/**
 * @brief The TwoStagePoCLModulePassManager class for running the full pipeline
 *
 * the full pipeline of PoCL LLVM passes looks like this:
 *  <pocl passes, LLVM optimization, pocl passes, LLVM optimization>
 *
 * this is how it worked with the old PM; however, running
 * all of these with a single PassManager (with new PM) leads
 * to crashes and/or failing tests. This class splits the execution of
 * the pipeline into two halfs (stages) so both contain some pocl-passes plus
 * the LLVM optimization passes, and uses a separate PassManager
 * instance for both (see class PoCLModulePassManager).
 * @note see also comment for SEPARATE_OPTIMIZATION_FROM_POCL_PASSES macro,
 * which (if enabled) further separates optimization pipeline into its own PM
 */
class TwoStagePoCLModulePassManager {
  PoCLModulePassManager Stage1;
  PoCLModulePassManager Stage2;
#ifndef PER_STAGE_TARGET_MACHINE
  std::unique_ptr<llvm::TargetMachine> Machine;
#endif
public:
  TwoStagePoCLModulePassManager() = default;
  llvm::Error build(cl_device_id Dev, const std::string &Stage1Pipeline,
                    unsigned Stage1OLevel, unsigned Stage1SLevel,
                    const std::string &Stage2Pipeline,
                    unsigned Stage2OLevel, unsigned Stage2SLevel);
  void run(llvm::Module &Bitcode);
};

llvm::Error TwoStagePoCLModulePassManager::build(
    cl_device_id Dev, const std::string &Stage1Pipeline, unsigned Stage1OLevel,
    unsigned Stage1SLevel, const std::string &Stage2Pipeline,
    unsigned Stage2OLevel, unsigned Stage2SLevel) {

  // Do not vectorize in the first round of (cleanup) optimizations to avoid
  // ending up with only vectorizing across the k-loops before the wi-loops have
  // been created. Let's leave the loop-interchange freedom to decide over which
  // loop to vectorize.
  bool Vectorize = false;

#ifndef PER_STAGE_TARGET_MACHINE
  Machine.reset(GetTargetMachine(Dev->llvm_target_triplet, Dev->llvm_cpu));
  TargetMachine *TMach = Machine.get();
#endif
  llvm::Error E1 =
      Stage1.build(Stage1Pipeline, Stage1OLevel, Stage1SLevel, Vectorize,
#ifndef PER_STAGE_TARGET_MACHINE
                   TMach,
#endif
                   Dev);
  if (E1)
    return E1;

  // Let's assume SPMD devices do their own vectorization at (SPIR-V) JIT time
  // if they see it beneficial.
  Vectorize = ((CurrentWgMethod == "loopvec" || CurrentWgMethod == "cbs") &&
               (!Dev->spmd));

  return Stage2.build(Stage2Pipeline, Stage2OLevel, Stage2SLevel, Vectorize,
#ifndef PER_STAGE_TARGET_MACHINE
                      TMach,
#endif
                      Dev);
}

void TwoStagePoCLModulePassManager::run(llvm::Module &Bitcode) {
  Stage1.run(Bitcode);
  Stage2.run(Bitcode);
}


enum class PassType {
  Module,
  CGSCC,
  Function,
  Loop,
};

// for legacy PM, add each Pass to the pass pipeline;
// for new PM, add them with proper nesting...  X(Y(...))
static void addPass(std::vector<std::string> &Passes, std::string PassName,
                    PassType T = PassType::Function) {
  std::string Temp;
  switch (T) {
  case PassType::Module:
    Passes.push_back(PassName);
    break;
  case PassType::CGSCC:
    Temp = "cgscc(" + PassName + ")";
    Passes.push_back(Temp);
    break;
  case PassType::Function:
    Temp = "function(" + PassName + ")";
    Passes.push_back(Temp);
    break;
  case PassType::Loop:
    Temp = "function(loop(" + PassName + "))";
    Passes.push_back(Temp);
    break;
  default:
    POCL_MSG_ERR("pocl_llvm_wg: addPass(): unknown pass type\n");
  }
}

// for legacy PM, add each Analysis to the pass pipeline like a normal Pass;
// for new PM, add them with proper nesting & analysis wrapper: X(Y(Require<>))
static void addAnalysis(std::vector<std::string> &Passes, std::string PassName,
                        PassType T = PassType::Function) {
  std::string Temp;
  PassName = "require<" + PassName + ">";
  switch (T) {
  case PassType::Module:
    Passes.push_back(PassName);
    break;
  case PassType::CGSCC:
    Temp = "cgscc(" + PassName + ")";
    Passes.push_back(Temp);
    break;
  case PassType::Function:
    Temp = "function(" + PassName + ")";
    Passes.push_back(Temp);
    break;
  case PassType::Loop:
    Temp = "function(loop(" + PassName + "))";
    Passes.push_back(Temp);
    break;
  default:
    POCL_MSG_ERR("pocl_llvm_wg: addPass(): unknown analysis type\n");
  }
}

// add the first part of the PoCL passes (up until 1st optimization in old PM)
// for old PM, also adds optimizations; for new PM it's handled separately
static void addStage1PassesToPipeline(cl_device_id Dev,
                                      std::vector<std::string> &Passes) {
  /* The kernel compiler passes to run, in order.

     Some notes about the kernel compiler phase ordering constraints:

     -mem2reg first because we get unoptimized output from Clang where all
     variables are allocas. Avoid context saving the allocas and make them
     more readable by calling -mem2reg at the beginning.

     -implicit-cond-barriers after -implicit-loop-barriers because the latter
     can inject barriers to loops inside conditional regions after which the
     peeling should be avoided by injecting the implicit conditional barriers.

     -loop-barriers, -barriertails, and -barriers should be ran after the
     implicit barrier injection passes so they "normalize" the implicit
     barriers also.

     -phistoallocas before -workitemloops as otherwise it cannot inject context
     restore code (PHIs need to be at the beginning of the BB and so one cannot
     context restore them with non-PHI code if the value is needed in another
     PHI).

     -automatic-locals after inline and always-inline; if we have a kernel
     that calls a non-kernel, and the non-kernel uses an automatic local
     (= GlobalVariable in LLVM), the 'automatic-locals' will skip processing
     of the non-kernel function, and the kernel function appears to it as not
     having any locals. Therefore the local variable remains a GV instead of
     being transformed into a kernel argument. This can lead to surprising
     result, as the final object ELF will contain a static variable, so the
     program will work with single-threaded execution, but multiple CPU
     threads will overwrite the static variable and produce garbage results.

     -optimize-wi-gvars after flatten-globals & always-inline passes
  */

  // NOTE: if you add a new PoCL pass here,
  // don't forget to register it in registerPassBuilderPasses
  addPass(Passes, "fix-min-legal-vec-size", PassType::Module);
  addPass(Passes, "inline-kernels");
  if (Dev->run_sanitize_divrem_pass)
    addPass(Passes, "sanitize-ub-of-div-rem");

  addPass(Passes, "handle-samplers");
  addPass(Passes, "infer-address-spaces");
  addPass(Passes, "mem2reg");
  addAnalysis(Passes, "domtree");
  addAnalysis(Passes, "workitem-handler-chooser");

  if (Dev->spmd) {
    addPass(Passes, "flatten-inline-all", PassType::Module);
  } else {
    addPass(Passes, "flatten-globals", PassType::Module);
    addPass(Passes, "flatten-barrier-subs", PassType::Module);
  }

  addPass(Passes, "always-inline", PassType::Module);

  // both of these must be done AFTER inlining, see note above
  addPass(Passes, "automatic-locals", PassType::Module);

  // Handle UnreachableInsts by converting them to returns or just deleting
  // them. Julia expects graceful handling of UIs with printouts before them. We
  // should convert the UIs in the input here otherwise optimizers will remove
  // them.
  if (!Dev->spmd)
    addPass(Passes, "unreachables-to-returns");

  // must come AFTER flatten-globals & always-inline
  addPass(Passes, "optimize-wi-gvars");

  // It should be now safe to run -O3 over the single work-item kernel
  // as the barrier has the attributes preventing illegal motions and
  // duplication. Let's do it to clean up the code for later Passes.
  // Especially the WI context structures get needlessly bloated in case there
  // is dead code lying around.
  // the optimization for new PM is handled separately
  // addPass(Passes, "STANDARD_OPTS");
}

// add the second part of the PoCL passes (after 1st, up to 2nd optimization in old PM)
// for old PM, also adds optimizations; for new PM it's handled separately
static void addStage2PassesToPipeline(cl_device_id Dev,
                                      std::vector<std::string> &Passes) {

  // NOTE: if you add a new PoCL pass here,
  // don't forget to register it in registerPassBuilderPasses
  if (!Dev->spmd) {
    addPass(Passes, "simplifycfg");
    addPass(Passes, "loop-simplify");

    // ...we have to call UTR again here because some optimizations in LLVM
    // might generate UIs.
    if (!Dev->spmd)
      addPass(Passes, "unreachables-to-returns");

    // required for OLD PM
    addAnalysis(Passes, "workitem-handler-chooser");
    addAnalysis(Passes, "pocl-vua");

    // Run lcssa explicitly to ensure it has generated its lcssa phis before
    // we break them in phistoallocas. This is an intermediate solution while
    // working towards processing unoptimized Clang output.
    addPass(Passes, "lcssa");
    addPass(Passes, "phistoallocas");
    addPass(Passes, "isolate-regions");

    // NEW PM requires WIH & VUA analyses here,
    // but they should not be invalidated by previous passes
    addPass(Passes, "implicit-loop-barriers", PassType::Loop);

    // implicit-cond-barriers handles barriers inside conditional
    // basic blocks (basically if...elses). It tries to minimize the
    // part ending up in the parallel region that is conditional by
    // isolating the branching condition (which must be uniform,
    // otherwise the end result is undefined according to barrier rules),
    // to minimize the impact of "work-item peeling" (* to describe).
    addPass(Passes, "implicit-cond-barriers");

    // loop-barriers adds implicit barriers to handle b-loops by isolating the
    // loop body from the loop construct. It also tries to make non b-loops
    // "isolated" in a way to produce the wiloop strictly around it, making
    // things nice for LLVM standard loop analysis (loop-interchange and
    // loopvec at least).
    addPass(Passes, "loop-barriers", PassType::Loop);

    addPass(Passes, "barriertails");
    addPass(Passes, "canon-barriers");
    addPass(Passes, "isolate-regions");

    // required for OLD PM
    addAnalysis(Passes, "wi-aa");
    addAnalysis(Passes, "workitem-handler-chooser");
    addAnalysis(Passes, "pocl-vua");

#if 0
    // use PoCL's own print-module pass
    // note: the "before" is an option given to the PoclCFGPrinter instance;
    // it will be used as a prefix to the dot files ("PREFIX_kernel.dot")
    addPass(Passes, "print<pocl-cfg;before>", PassType::Module);
#endif

    // subcfgformation (for CBS) before workitemloops, as wiloops creates the
    // loops for kernels without barriers, but after the transformation the
    // kernel looks like it has barriers, so subcfg would do its thing.
    addPass(Passes, "subcfgformation");

    // subcfgformation before workitemloops, as wiloops creates the loops for
    // kernels without barriers, but after the transformation the kernel looks
    // like it has barriers, so subcfg would do its thing.
    addPass(Passes, "workitemloops");
    // Remove the (pseudo) barriers.   They have no use anymore due to the
    // work-item loop control taking care of them.
    addPass(Passes, "remove-barriers");
  }

  // verify & print the module
#if 0
  addPass(Passes, "verify", PassType::Module);
  addPass(Passes, "print", PassType::Module);
#endif


  // Add the work group launcher functions and privatize the pseudo variable
  // (local id) accesses. We have to do this late because we rely on aggressive
  // inlining to expose the _{local,group}_id accesses which will be replaced
  // with context struct accesses. TODO: A cleaner and a more robust way would
  // be to add hidden context struct parameters to the builtins that need the
  // context data and fix the calls early.
  if (Dev->run_workgroup_pass) {
    addPass(Passes, "workgroup", PassType::Module);
    addPass(Passes, "always-inline", PassType::Module);
  }

  // Attempt to move all allocas to the entry block to avoid the need for
  // dynamic stack which is problematic for some architectures.
  addPass(Passes, "allocastoentry");

  // Convert variables back to PHIs to clean up loop structures to enable the
  // LLVM standard loop analysis.
  addPass(Passes, "mem2reg");

  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running
  // the standard LLVM optimizations.
  addPass(Passes, "simplifycfg");

  // the optimization for new PM is handled separately
  // addPass(Passes, "STANDARD_OPTS");

  // Due to unfortunate phase-ordering problems with store sinking,
  // loop deletion does not always apply when executing -O3 only
  // once. Cherry pick the optimization to rerun here.
  // addPass(Passes, "loop-deletion");
  // addPass(Passes, "remove-barriers");
}

// old PM uses a vector of strings directly; new PM requires a single string
// vector.join(",")
static std::string convertPassesToPipelineString(const std::vector<std::string> &Passes) {
  std::string Pipeline;
  for (auto It = Passes.begin(); It != Passes.end(); ++It) {
    Pipeline.append(*It);
    Pipeline.append(",");
  }
  if (!Pipeline.empty())
    Pipeline.pop_back();
  return Pipeline;
}

static bool runKernelCompilerPasses(cl_device_id Device, llvm::Module &Mod,
                                    bool Optimize) {

  TwoStagePoCLModulePassManager PM;
  std::vector<std::string> Passes1;
  addStage1PassesToPipeline(Device, Passes1);
  std::string P1 = convertPassesToPipelineString(Passes1);
  std::vector<std::string> Passes2;
  addStage2PassesToPipeline(Device, Passes2);
  std::string P2 = convertPassesToPipelineString(Passes2);

  Error E = PM.build(Device, P1, Optimize ? 1 : 0, 0, P2, Optimize ? 3 : 0, 0);
  if (E) {
    std::cerr << "LLVM: failed to create compilation pipeline";
    return false;
  }

  PM.run(Mod);
  return true;
}

void pocl_destroy_llvm_module(void *modp, cl_context ctx) {

  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  llvm::Module *mod = (llvm::Module *)modp;
  if (mod) {
    delete mod;
    --llvm_ctx->number_of_IRs;
  }
}

#ifdef ENABLE_SPIRV
namespace pocl {
class ProgramWithContext {

  llvm::LLVMContext LLVMCtx;
  std::unique_ptr<llvm::Module> ProgramBC;
  std::unique_ptr<llvm::Module> ProgramGVarsNonKernelsBC;
  std::mutex Lock;
  unsigned Num = 0;

public:

  bool init(const char *ProgramBcBytes,
            size_t ProgramBcSize,
            char* LinkinOutputBCPath) {
    Num = 0;
    llvm::Module *P = parseModuleIRMem(ProgramBcBytes, ProgramBcSize, &LLVMCtx);
    if (P == nullptr)
      return false;
    ProgramBC.reset(P);

    ProgramGVarsNonKernelsBC.reset(
        new llvm::Module(llvm::StringRef("program_gvars.bc"), LLVMCtx));

    ProgramGVarsNonKernelsBC->setTargetTriple(ProgramBC->getTargetTriple());
    ProgramGVarsNonKernelsBC->setDataLayout(ProgramBC->getDataLayout());

    if (!moveProgramScopeVarsOutOfProgramBc(&LLVMCtx, ProgramBC.get(),
                                            ProgramGVarsNonKernelsBC.get(),
                                            SPIR_ADDRESS_SPACE_LOCAL))
      return false;

    pocl_cache_tempname(LinkinOutputBCPath, ".bc", NULL);
    int r = pocl_write_module(ProgramGVarsNonKernelsBC.get(),
                              LinkinOutputBCPath);
    if (r != 0) {
      POCL_MSG_ERR("ProgramWithContext->init: failed to write module\n");
      return false;
    }

    if (verifyIR()) {
      std::string ErrorLog;
      llvm::raw_string_ostream Errs(ErrorLog);
      if (llvm::verifyModule(*ProgramGVarsNonKernelsBC.get(), &Errs)) {
        POCL_MSG_ERR("Failed to verify Program GVars Module:\n%s\n",
                     ErrorLog.c_str());
        return false;
      }
    }

    return true;
  }

  bool getBitcodeForKernel(const char *KernelName, std::string &OutputBitcode,
                           std::string *BuildLog) {
    std::lock_guard<std::mutex> LockGuard(Lock);

    // Create an empty Module and copy only the kernel+callgraph from
    // program.bc.
    std::unique_ptr<llvm::Module> KernelBC(
        new llvm::Module(llvm::StringRef("parallel_bc"), LLVMCtx));

    KernelBC->setTargetTriple(ProgramBC->getTargetTriple());
    KernelBC->setDataLayout(ProgramBC->getDataLayout());

    copyKernelFromBitcode(KernelName, KernelBC.get(), ProgramBC.get(), nullptr);

    if (verifyIR()) {
      llvm::raw_string_ostream Errs(*BuildLog);
      if (llvm::verifyModule(*KernelBC.get(), &Errs)) {
        POCL_MSG_ERR("Failed to verify Kernel Module:\n%s\n",
                     BuildLog->c_str());
        BuildLog->append("Failed to verify Kernel Module\n");
        return false;
      }
    }

    writeModuleIRtoString(KernelBC.get(), OutputBitcode);
    return true;
  }
};
} // namespace pocl

void *pocl_llvm_create_context_for_program(char *ProgramBcContent,
                                           size_t ProgramBcSize,
                                           char **LinkinSpirvContent,
                                           uint64_t *LinkinSpirvSize,
                                           pocl_version_t TargetVersion) {
  assert(ProgramBcContent);
  assert(ProgramBcSize > 0);

  char TempBitcodePath[POCL_MAX_PATHNAME_LENGTH];

  pocl::ProgramWithContext *P = new pocl::ProgramWithContext;
  // parse the program's bytes into a llvm::Module
  if (P == nullptr ||
      !P->init(ProgramBcContent, ProgramBcSize, TempBitcodePath)) {
    if (P)
      delete P;
    POCL_MSG_ERR("failed to create program for context");
    return nullptr;
  }

  std::string BuildLog;
  if (pocl_convert_bitcode_to_spirv2(
          nullptr, ProgramBcContent, ProgramBcSize, &BuildLog,
          "all", // TODO SPIRV exts
          nullptr, LinkinSpirvContent, LinkinSpirvSize, TargetVersion) != 0) {
    delete P;
    POCL_MSG_ERR("failed to create program for context, log:%s\n",
                 BuildLog.c_str());
    return nullptr;
  }

  return (void *)P;
}

void pocl_llvm_release_context_for_program(void *ProgCtx) {
  if (ProgCtx == nullptr)
    return;
  pocl::ProgramWithContext *P = (pocl::ProgramWithContext *)ProgCtx;
  delete P;
}

// extract SPIRV of a single Kernel from a program
int pocl_llvm_extract_kernel_spirv(
    void *ProgCtx, const char *KernelName, void *BuildLogStr,
    char **SpirvContent, uint64_t *SpirvSize, pocl_version_t TargetVersion) {

  POCL_MEASURE_START(extractKernel);

  std::string *BuildLog = (std::string *)BuildLogStr;

  std::string OutputBitcode;

  pocl::ProgramWithContext *P = (pocl::ProgramWithContext *)ProgCtx;
  if (!P->getBitcodeForKernel(KernelName, OutputBitcode, BuildLog)) {
    return -1;
  }

  int R = pocl_convert_bitcode_to_spirv2(
      nullptr, OutputBitcode.data(), OutputBitcode.size(), &BuildLog,
      "all",   // TODO SPIRV Exts
      nullptr, // SpirvOutputPath
      SpirvContent, SpirvSize, TargetVersion);

  POCL_MEASURE_FINISH(extractKernel);
  return R;
}

#endif // ENABLE_SPIRV

static int
pocl_llvm_run_pocl_passes(llvm::Module *Bitcode,
                          _cl_command_run *RunCommand, // optional
                          [[maybe_unused]] llvm::LLVMContext *LLVMContext,
                          PoclLLVMContextData *PoclCtx, cl_program Program,
                          cl_kernel Kernel, // optional
                          cl_device_id Device, int Specialize) {
  // Set to true to generate a global offset 0 specialized WG function.
  bool WGAssumeZeroGlobalOffset;
  // If set to true, the next 3 parameters define the local size to specialize
  // for.
  bool WGDynamicLocalSize;
  size_t WGLocalSizeX;
  size_t WGLocalSizeY;
  size_t WGLocalSizeZ;
  // If set to non-zero, assume each grid dimension is at most this
  // work-items wide.
  size_t WGMaxGridDimWidth;

  // Set the specialization properties.
  if (Specialize) {
    assert(RunCommand);
    WGLocalSizeX = RunCommand->pc.local_size[0];
    WGLocalSizeY = RunCommand->pc.local_size[1];
    WGLocalSizeZ = RunCommand->pc.local_size[2];
    WGDynamicLocalSize =
        WGLocalSizeX == 0 && WGLocalSizeY == 0 && WGLocalSizeZ == 0;
    WGAssumeZeroGlobalOffset = RunCommand->pc.global_offset[0] == 0 &&
                               RunCommand->pc.global_offset[1] == 0 &&
                               RunCommand->pc.global_offset[2] == 0;
    // Compile a smallgrid version or a generic one?
    if (RunCommand->force_large_grid_wg_func ||
        pocl_cmd_max_grid_dim_width(RunCommand) >=
            Device->grid_width_specialization_limit) {
      WGMaxGridDimWidth = 0; // The generic / large / unlimited size one.
    } else {
      // Limited grid dimension width by the device specific limit.
      WGMaxGridDimWidth = Device->grid_width_specialization_limit;
    }
  } else {
    WGDynamicLocalSize = true;
    WGLocalSizeX = WGLocalSizeY = WGLocalSizeZ = 0;
    WGAssumeZeroGlobalOffset = false;
    WGMaxGridDimWidth = 0;
  }

  if (Device->device_aux_functions) {
    std::string concat;
    const char **tmp = Device->device_aux_functions;
    while (*tmp != nullptr) {
      concat.append(*tmp);
      ++tmp;
      if (*tmp)
        concat.append(";");
    }
    setModuleStringMetadata(Bitcode, "device_aux_functions", concat.c_str());
  }

  setModuleIntMetadata(Bitcode, "device_address_bits", Device->address_bits);
  setModuleBoolMetadata(Bitcode, "device_arg_buffer_launcher",
                        Device->arg_buffer_launcher);
  setModuleBoolMetadata(Bitcode, "device_grid_launcher", Device->grid_launcher);
  setModuleBoolMetadata(Bitcode, "device_is_spmd", Device->spmd);

  if (Device->native_vector_width_in_bits)
    setModuleIntMetadata(Bitcode, "device_native_vec_width",
                         Device->native_vector_width_in_bits);

  if (Kernel != nullptr)
    setModuleStringMetadata(Bitcode, "KernelName", Kernel->name);

  setModuleIntMetadata(Bitcode, "WGMaxGridDimWidth", WGMaxGridDimWidth);
  setModuleIntMetadata(Bitcode, "WGLocalSizeX", WGLocalSizeX);
  setModuleIntMetadata(Bitcode, "WGLocalSizeY", WGLocalSizeY);
  setModuleIntMetadata(Bitcode, "WGLocalSizeZ", WGLocalSizeZ);
  setModuleBoolMetadata(Bitcode, "WGDynamicLocalSize", WGDynamicLocalSize);
  setModuleBoolMetadata(Bitcode, "WGAssumeZeroGlobalOffset",
                        WGAssumeZeroGlobalOffset);

  setModuleIntMetadata(Bitcode, "device_global_as_id", Device->global_as_id);
  setModuleIntMetadata(Bitcode, "device_local_as_id", Device->local_as_id);
  setModuleIntMetadata(Bitcode, "device_constant_as_id",
                       Device->constant_as_id);
  setModuleIntMetadata(Bitcode, "device_args_as_id", Device->args_as_id);
  setModuleIntMetadata(Bitcode, "device_context_as_id", Device->context_as_id);

  setModuleBoolMetadata(Bitcode, "device_side_printf",
                        Device->device_side_printf);
  setModuleBoolMetadata(Bitcode, "device_alloca_locals",
                        Device->device_alloca_locals);
  setModuleIntMetadata(Bitcode, "device_autolocals_to_args",
                       (unsigned long)Device->autolocals_to_args);

  setModuleIntMetadata(Bitcode, "device_max_witem_dim",
                       Device->max_work_item_dimensions);
  setModuleIntMetadata(Bitcode, "device_max_witem_sizes_0",
                       Device->max_work_item_sizes[0]);
  setModuleIntMetadata(Bitcode, "device_max_witem_sizes_1",
                       Device->max_work_item_sizes[1]);
  setModuleIntMetadata(Bitcode, "device_max_witem_sizes_2",
                       Device->max_work_item_sizes[2]);

  std::string Opts;
  if (Program->compiler_options)
    Opts.assign(Program->compiler_options);
  bool Optimize = (Opts.find("-cl-opt-disable") == std::string::npos);
#ifdef DUMP_LLVM_PASS_TIMINGS
  llvm::TimePassesIsEnabled = true;
#endif
  POCL_MEASURE_START(llvm_workgroup_ir_func_gen);
  runKernelCompilerPasses(Device, *Bitcode, Optimize);
  POCL_MEASURE_FINISH(llvm_workgroup_ir_func_gen);
#ifdef DUMP_LLVM_PASS_TIMINGS
  llvm::reportAndResetTimings();
#endif

  // Print loop vectorizer remarks if enabled.
  if (pocl_get_bool_option("POCL_VECTORIZER_REMARKS", 0) == 1) {
    std::cerr << getDiagString(PoclCtx);
  }

  return 0;
}

int pocl_llvm_generate_workgroup_function_nowrite(
    unsigned DeviceI, cl_device_id Device, cl_kernel Kernel,
    _cl_command_node *Command, void **Output, int Specialize) {

  _cl_command_run *RunCommand = &Command->command.run;
  cl_program Program = Kernel->program;
  cl_context ctx = Program->context;
  PoclLLVMContextData *PoCLLLVMContext =
      (PoclLLVMContextData *)ctx->llvm_context_data;
  llvm::LLVMContext *LLVMContext = PoCLLLVMContext->Context;
  llvm::Module *ParallelBC = nullptr;
  PoclCompilerMutexGuard lockHolder(&PoCLLLVMContext->Lock);

#ifdef DEBUG_POCL_LLVM_API
  printf("### calling generate_WG_function for kernel %s local_x %zu "
         "local_y %zu local_z %zu parallel_filename: %s\n",
         kernel->name, local_x, local_y, local_z, parallel_bc_path);
#endif

  llvm::Module *ProgramBC = (llvm::Module *)Program->llvm_irs[DeviceI];

  // Create an empty Module and copy only the kernel+callgraph from
  // program.bc.
  ParallelBC = new llvm::Module(StringRef("parallel_bc"), *LLVMContext);

  ParallelBC->setTargetTriple(ProgramBC->getTargetTriple());
  ParallelBC->setDataLayout(ProgramBC->getDataLayout());

  copyKernelFromBitcode(Kernel->name, ParallelBC, ProgramBC,
                        Device->device_aux_functions);

  int res = pocl_llvm_run_pocl_passes(ParallelBC, RunCommand, LLVMContext,
                                      PoCLLLVMContext, Program, Kernel, Device,
                                      Specialize);

  std::string FinalizerCommand =
      pocl_get_string_option("POCL_BITCODE_FINALIZER", "");
  if (!FinalizerCommand.empty()) {
    // Run a user-defined command on the final bitcode.
    char TempParallelBCFileName[POCL_MAX_PATHNAME_LENGTH];
    int FD = -1, Err = 0;

    Err = pocl_mk_tempname(TempParallelBCFileName, "/tmp/pocl-parallel", ".bc",
                           &FD);
    POCL_RETURN_ERROR_ON((Err != 0), CL_FAILED,
                         "Failed to create "
                         "temporary file %s\n",
                         TempParallelBCFileName);
    Err = pocl_write_module((char *)ParallelBC, TempParallelBCFileName);
    POCL_RETURN_ERROR_ON((Err != 0), CL_FAILED,
                         "Failed to write bitcode "
                         "into temporary file %s\n",
                         TempParallelBCFileName);

    std::string Command = std::regex_replace(
        FinalizerCommand, std::regex(R"(%\(bc\))"), TempParallelBCFileName);
    Err = system(Command.c_str());
    POCL_RETURN_ERROR_ON((Err != 0), CL_FAILED,
                         "Failed to execute "
                         "bitcode finalizer\n");

    llvm::Module *NewBitcode =
        parseModuleIR(TempParallelBCFileName, LLVMContext);
    POCL_RETURN_ERROR_ON((NewBitcode == nullptr), CL_FAILED,
                         "failed to parse bitcode from finalizer\n");
    delete ParallelBC;
    ParallelBC = NewBitcode;
  }

  assert(Output != NULL);
  if (res == 0) {
    *Output = (void *)ParallelBC;
    ++PoCLLLVMContext->number_of_IRs;
  } else {
    *Output = nullptr;
  }

  return res;
}

int pocl_llvm_run_passes_on_program(cl_program Program, unsigned DeviceI) {

  llvm::Module *ProgramBC = (llvm::Module *)Program->llvm_irs[DeviceI];
  cl_device_id Device = Program->devices[DeviceI];
  cl_context ctx = Program->context;
  PoclLLVMContextData *PoCLLLVMContext =
      (PoclLLVMContextData *)ctx->llvm_context_data;
  llvm::LLVMContext *LLVMContext = PoCLLLVMContext->Context;
  PoclCompilerMutexGuard lockHolder(&PoCLLLVMContext->Lock);

  return pocl_llvm_run_pocl_passes(ProgramBC,
                                   nullptr, // RunCommand,
                                   LLVMContext, PoCLLLVMContext,
                                   Program,
                                   nullptr, // Kernel,
                                   Device,
                                   0); // Specialize
}

int pocl_llvm_generate_workgroup_function(unsigned DeviceI, cl_device_id Device,
                                          cl_kernel Kernel,
                                          _cl_command_node *Command,
                                          int Specialize) {
  cl_context ctx = Kernel->context;
  void *Module = NULL;

  char ParallelBCPath[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_work_group_function_path(ParallelBCPath, Kernel->program, DeviceI,
                                      Kernel, Command, Specialize);

  if (pocl_exists(ParallelBCPath))
    return CL_SUCCESS;

  char FinalBinaryPath[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_final_binary_path(FinalBinaryPath, Kernel->program, DeviceI,
                               Kernel, Command, Specialize);

  if (pocl_exists(FinalBinaryPath))
    return CL_SUCCESS;

  int Error = pocl_llvm_generate_workgroup_function_nowrite(
      DeviceI, Device, Kernel, Command, &Module, Specialize);
  if (Error)
    return Error;

  Error = pocl_cache_write_kernel_parallel_bc(Module, Kernel->program, DeviceI,
                                              Kernel, Command, Specialize);

  if (Error) {
    POCL_MSG_ERR("pocl_cache_write_kernel_parallel_bc() failed with %i\n",
                 Error);
    return Error;
  }

  pocl_destroy_llvm_module(Module, ctx);
  return Error;
}

/* Reads LLVM IR module from program->binaries[i], if prog_data->llvm_ir is
 * NULL */
int pocl_llvm_read_program_llvm_irs(cl_program program, unsigned device_i,
                                    const char *program_bc_path) {
  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);
  cl_device_id dev = program->devices[device_i];

  if (program->llvm_irs[device_i] != nullptr)
    return CL_SUCCESS;

  llvm::Module *M;
  if (program->binaries[device_i])
    M = parseModuleIRMem((char *)program->binaries[device_i],
                         program->binary_sizes[device_i], llvm_ctx->Context);
  else {
    // TODO
    assert(program_bc_path);
    M = parseModuleIR(program_bc_path, llvm_ctx->Context);
  }
  assert(M);
  program->llvm_irs[device_i] = M;
  if (dev->run_program_scope_variables_pass)
    parseModuleGVarSize(program, device_i, M);
  ++llvm_ctx->number_of_IRs;
  return CL_SUCCESS;
}

int pocl_llvm_link_multiple_modules(cl_program program, unsigned device_i,
                                    const char *OutputBCPath,
                                    void **LLVMIRBinaries, size_t NumBinaries) {
  POCL_RETURN_ERROR_COND((LLVMIRBinaries == nullptr), CL_LINK_PROGRAM_FAILURE);

  pocl_llvm_free_llvm_irs(program, device_i);

  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);
  llvm::Module *Dest = new llvm::Module("linked_mod", *llvm_ctx->Context);

  for (cl_uint i = 0; i < NumBinaries; ++i) {
    llvm::Module *Mod = (llvm::Module *)LLVMIRBinaries[i];
    assert(Mod);
    assert(&Mod->getContext() == llvm_ctx->Context);
    if (llvm::Linker::linkModules(*Dest, llvm::CloneModule(*Mod))) {
      delete Dest;
      return CL_LINK_PROGRAM_FAILURE;
    }
  }
  program->llvm_irs[device_i] = Dest;
  return pocl_write_module(Dest, OutputBCPath);
}

int pocl_llvm_recalculate_gvar_sizes(cl_program Program, unsigned DeviceI) {
  std::string ErrLog;
  std::set<llvm::GlobalVariable *> GVarSet;
  cl_device_id Dev = Program->devices[DeviceI];

  assert(Program->llvm_irs[DeviceI] != nullptr);
  llvm::Module *M = (llvm::Module *)Program->llvm_irs[DeviceI];
  assert(Program->global_var_total_size != nullptr);
  Program->global_var_total_size[DeviceI] = 0;

  if (!pocl::areAllGvarsDefined(M, ErrLog, GVarSet, Dev->local_as_id)) {
    POCL_MSG_ERR("Not all GVars are defined: \n%s\n", ErrLog.c_str());
    return CL_FAILED;
  }
  std::map<llvm::GlobalVariable *, uint64_t> GVarOffsets;
  size_t TotalSize =
      pocl::calculateGVarOffsetsSizes(M->getDataLayout(), GVarOffsets, GVarSet);
  Program->global_var_total_size[DeviceI] = TotalSize;
  return CL_SUCCESS;
}

void pocl_llvm_free_llvm_irs(cl_program program, unsigned device_i) {
  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  if (program->llvm_irs[device_i]) {
    llvm::Module *mod = (llvm::Module *)program->llvm_irs[device_i];
    delete mod;
    --llvm_ctx->number_of_IRs;
    program->llvm_irs[device_i] = nullptr;
  }
}

static TargetLibraryInfoImpl *initPassManagerForCodeGen(legacy::PassManager &PM,
                                                        const char* TTriple,
                                                        cl_device_type DevType) {
  assert(TTriple);
  llvm::Triple DevTriple(TTriple);
  llvm::TargetLibraryInfoWrapperPass *TLIPass = nullptr;
  TargetLibraryInfoImpl *TLII = nullptr;

#ifdef ENABLE_HOST_CPU_VECTORIZE_BUILTINS
  if (DevType == CL_DEVICE_TYPE_CPU) {
    TLII =
        llvm::driver::createTLII(DevTriple,
#ifdef ENABLE_HOST_CPU_VECTORIZE_LIBMVEC
                                 driver::VectorLibrary::LIBMVEC);
#endif
#ifdef ENABLE_HOST_CPU_VECTORIZE_SLEEF
                                 driver::VectorLibrary::SLEEF);
#endif
#ifdef ENABLE_HOST_CPU_VECTORIZE_SVML
                                 driver::VectorLibrary::SVML);
#endif
    TLIPass = new TargetLibraryInfoWrapperPass(*TLII);
  } else
#endif
  {
    TLIPass = new TargetLibraryInfoWrapperPass(DevTriple);
  }

  PM.add(TLIPass);
  return TLII;
}

/* Run LLVM codegen on input file (parallel-optimized).
 * modp = llvm::Module* of parallel.bc
 * Output native object file (<kernel>.so.o). */
int pocl_llvm_codegen2(const char* TTriple, const char* MCPU,
                       const char *Features, cl_device_type DevType,
                       pocl_lock_t *Lock, void *Modp, int EmitAsm,
                       int EmitObj, char **Output, uint64_t *OutputSize) {

  PoclCompilerMutexGuard LockHolder(Lock);

  llvm::Module *Input = (llvm::Module *)Modp;
  assert(Input);
  *Output = nullptr;
  std::unique_ptr<llvm::TargetLibraryInfoImpl> TLIIPtr;

  std::unique_ptr<llvm::TargetMachine> TM(GetTargetMachine(TTriple, MCPU, Features));
  llvm::TargetMachine *Target = TM.get();

  // First try direct object code generation from LLVM, if supported by the
  // LLVM backend for the target.
  bool LLVMGeneratesObjectFiles = true;

  SmallVector<char, 4096> Data;
  llvm::raw_svector_ostream SOS(Data);
  bool cannotEmitFile;

  assert(EmitObj || EmitAsm);

  if (EmitObj) {
    legacy::PassManager PMObj;
    TLIIPtr.reset(initPassManagerForCodeGen(PMObj, TTriple, DevType));

    cannotEmitFile = Target->addPassesToEmitFile(PMObj, SOS, nullptr,
  #if LLVM_MAJOR < 18
                                                 llvm::CGFT_ObjectFile);
  #else
                                                 llvm::CodeGenFileType::
                                                     ObjectFile);
  #endif
    LLVMGeneratesObjectFiles = !cannotEmitFile;

    if (LLVMGeneratesObjectFiles) {
      POCL_MSG_PRINT_LLVM("Generating an object file directly.\n");
  #ifdef DUMP_LLVM_PASS_TIMINGS
      llvm::TimePassesIsEnabled = true;
  #endif
      PMObj.run(*Input);
  #ifdef DUMP_LLVM_PASS_TIMINGS
      llvm::reportAndResetTimings();
  #endif
      auto O = SOS.str(); // flush
      const char *Cstr = O.data();
      size_t S = O.size();
      *Output = (char *)malloc(S);
      *OutputSize = S;
      memcpy(*Output, Cstr, S);
      return 0;
    } else {
      if (!EmitAsm) {
        POCL_MSG_ERR("llvm_codegen: The target doesn't support "
                     "obj emission & asm emission not permitted\n");
        return -1;
      }
    }
  }

  if (EmitAsm) {
    legacy::PassManager PMAsm;
    TLIIPtr.reset(initPassManagerForCodeGen(PMAsm, TTriple, DevType));

    POCL_MSG_PRINT_LLVM("Generating assembly text.\n");

    // The LLVM target does not implement support for emitting object file directly.
    // Have to emit the text first and then call the assembler from the command line
    // to produce the binary.

    if (Target->addPassesToEmitFile(PMAsm, SOS, nullptr,
  #if LLVM_MAJOR < 18
                                    llvm::CGFT_AssemblyFile)
  #else
                                    llvm::CodeGenFileType::AssemblyFile)
  #endif
    ) {
      POCL_MSG_ERR(
          "llvm_codegen: The target supports neither obj nor asm emission!");
      return -1;
    }
  #ifdef DUMP_LLVM_PASS_TIMINGS
    llvm::TimePassesIsEnabled = true;
  #endif
    // This produces the assembly text:
    PMAsm.run(*Input);
  #ifdef DUMP_LLVM_PASS_TIMINGS
    llvm::reportAndResetTimings();
  #endif
  }

  if (!EmitObj) {
    // return generated Asm to the caller
    auto O = SOS.str(); // flush
    const char *Cstr = O.data();
    size_t S = O.size();
    *Output = (char *)malloc(S);
    *OutputSize = S;
    memcpy(*Output, Cstr, S);
    return 0;
  } else {
    // Next call the target's assembler via the Toolchain API indirectly through
    // the Driver API.
    char AsmFileName[POCL_MAX_PATHNAME_LENGTH];
    char ObjFileName[POCL_MAX_PATHNAME_LENGTH];

    std::string AsmStr = SOS.str().str();
    pocl_cache_write_kernel_asmfile(AsmFileName, AsmStr.c_str(), AsmStr.size());
    pocl_cache_tempname(ObjFileName, OBJ_EXT, nullptr);

    std::string CpuFlag = (MCPU != nullptr)
                           ? (std::string(CLANG_MARCH_FLAG) + MCPU)
                           : "";

    const char *Args[] = {pocl_get_path("CLANG", CLANGCC),
                          AsmFileName,
                          "-c",
                          "-o",
                          ObjFileName,
                          MCPU ? CpuFlag.c_str() : nullptr,
                          nullptr};
    int Res = pocl_invoke_clang(TTriple, Args);
    if (Res == 0) {
      if (pocl_read_file(ObjFileName, Output, OutputSize)) {
        POCL_MSG_ERR("Could not read the object file.");
        return -1;
      }
    }
    if (pocl_remove(AsmFileName))
      POCL_MSG_ERR("failed to remove %s\n", AsmFileName);
    if (pocl_remove(ObjFileName))
      POCL_MSG_ERR("failed to remove %s\n", ObjFileName);
    return 0;
  }

  return -1;
}


int pocl_llvm_codegen(cl_device_id Device, cl_program Program,
                      const char *Features, void *Modp, int EmitAsm,
                      int EmitObj, char **Output, uint64_t *OutputSize) {

  cl_context Ctx = Program->context;
  PoclLLVMContextData *LLVMCtx = (PoclLLVMContextData *)Ctx->llvm_context_data;

  return pocl_llvm_codegen2 (Device->llvm_target_triplet, Device->llvm_cpu,
                            Features, Device->type, &LLVMCtx->Lock,
                            Modp, EmitAsm, EmitObj, Output, OutputSize);
}

void populateModulePM([[maybe_unused]] void *Passes, void *Module,
                      unsigned OptL, unsigned SizeL, bool Vectorize,
                      TargetMachine *TM) {

  PipelineTuningOptions PTO;

  // Let the loopvec decide when to unroll.
  PTO.LoopUnrolling = false;
  PTO.UnifiedLTO = false;
  PTO.LoopInterleaving = Vectorize;
  PTO.SLPVectorization = Vectorize;
  PTO.LoopVectorization = Vectorize;

  // Create the analysis managers.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PrintPassOptions PrintPassOpts;
  PassInstrumentationCallbacks PIC;
  llvm::LLVMContext Context; // for SI
  std::unique_ptr<StandardInstrumentations> SI;
  PrintPassOpts.Verbose = false;
  PrintPassOpts.SkipAnalyses = true;
  PrintPassOpts.Indent = true;
  SI.reset(new StandardInstrumentations(Context,
                                        enableDebugLogs(), // debug logging
                                        verifyIR(),        // verify each
                                        PrintPassOpts));
  SI->registerCallbacks(PIC, &MAM);

  PassBuilder PB(TM, PTO, std::nullopt, &PIC);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Opt constructior is private
  OptimizationLevel Opt;
  if (SizeL > 0)
    Opt = OptimizationLevel::Os;
  else {
    switch (OptL) {
    case 0:
      Opt = OptimizationLevel::O0;
      break;
    case 1:
      Opt = OptimizationLevel::O1;
      break;
    case 2:
      Opt = OptimizationLevel::O2;
      break;
    default:
    case 3:
      Opt = OptimizationLevel::O3;
      break;
    }
  }
  ModulePassManager MPM;
  if (Opt == OptimizationLevel::O0)
    MPM = PB.buildO0DefaultPipeline(Opt);
  else
    MPM = PB.buildPerModuleDefaultPipeline(Opt);
  if (Module) {
    llvm::Module *Mod = (llvm::Module *)Module;
    MPM.run(*Mod, MAM);
  }
}

/* vim: set ts=4 expandtab: */
