/* pocl_llvm_wg.cc: part of pocl LLVM API dealing with parallel.bc,
   optimization passes and codegen.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2019 Pekka Jääskeläinen
                 2023 Pekka Jääskeläinen / Intel Finland Oy

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
#if LLVM_VERSION_MAJOR < 16
#include <llvm/ADT/Triple.h>
#else
#include <llvm/TargetParser/Triple.h>
#endif
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/Support/Casting.h>
#if LLVM_VERSION_MAJOR < 14
#include <llvm/Support/TargetRegistry.h>
#else
#include <llvm/MC/TargetRegistry.h>
#endif
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
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Utils/Cloning.h>
// legacy PassManager is needed for CodeGen, even if new PM is used for IR
#include <llvm/IR/LegacyPassManager.h>
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#else
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#endif

#include "LLVMUtils.h"
POP_COMPILER_DIAGS

#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_spir.h"
#include "pocl_util.h"

#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "linker.h"

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

/* FIXME: these options should come from the cl_device, and
 * cl_program's options. */
static llvm::TargetOptions GetTargetOptions() {
  llvm::TargetOptions Options;
#ifdef HOST_FLOAT_SOFT_ABI
  Options.FloatABIType = FloatABI::Soft;
#else
  Options.FloatABIType = FloatABI::Hard;
#endif
  return Options;
}


// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine *GetTargetMachine(cl_device_id device) {

  std::string Error;
  Triple DevTriple(device->llvm_target_triplet);

  std::string MCPU = device->llvm_cpu ? device->llvm_cpu : "";

  const Target *TheTarget = TargetRegistry::lookupTarget("", DevTriple, Error);

  // In LLVM 3.4 and earlier, the target registry falls back to
  // the cpp backend in case a proper match was not found. In
  // that case simply do not use target info in the compilation
  // because it can be an off-tree target not registered at
  // this point (read: TCE).
  if (!TheTarget || TheTarget->getName() == std::string("cpp")) {
    return nullptr;
  }

  TargetMachine *TM = TheTarget->createTargetMachine(
      DevTriple.getTriple(), MCPU, StringRef(""), GetTargetOptions(),
      Reloc::PIC_, CodeModel::Small, CodeGenOpt::Aggressive);

  assert(TM != NULL && "llvm target has no targetMachine constructor");
  if (device->ops->init_target_machine)
    device->ops->init_target_machine(device->data, TM);

  return TM;
}


#if LLVM_MAJOR >= MIN_LLVM_NEW_PASSMANAGER
class PoclModulePassManager {
  // Create the analysis managers.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  ModulePassManager PM;
  PipelineTuningOptions PTO;
  std::unique_ptr<TargetLibraryInfoImpl> TLII;
  std::unique_ptr<StandardInstrumentations> SI;
#ifdef PER_STAGE_TARGET_MACHINE
  std::unique_ptr<llvm::TargetMachine> Machine;
#endif
#ifdef DEBUG_NEW_PASS_MANAGER
  PrintPassOptions PrintPassOpts;
  PassInstrumentationCallbacks PIC;
  llvm::LLVMContext Context; // for SI
#endif
  std::unique_ptr<PassBuilder> PassB;
  unsigned OptimizeLevel;
  unsigned SizeLevel;
  bool Vectorize;

public:
  PoclModulePassManager() = default;
  llvm::Error build(std::string PoclPipeline,
                    unsigned OLevel, unsigned SLevel,
#ifndef PER_STAGE_TARGET_MACHINE
                    TargetMachine *TM,
#endif
                    cl_device_id Dev);
  void run(llvm::Module &Bitcode);
};

llvm::Error PoclModulePassManager::build(std::string PoclPipeline,
                                         unsigned OLevel, unsigned SLevel,
#ifndef PER_STAGE_TARGET_MACHINE
                                         TargetMachine *TM,
#endif
                                         cl_device_id Dev) {

#ifdef PER_STAGE_TARGET_MACHINE
  Machine.reset(GetTargetMachine(Dev));
  TargetMachine *TM = Machine.get();
#endif

  PTO.LoopUnrolling = false;
#if LLVM_MAJOR > 16
  PTO.UnifiedLTO = false;
#endif
  // These need to be setup in addition to invoking the passes
  // to get the vectorizers initialized properly. Assume SPMD
  // devices do not want to vectorize intra work-item at this
  // stage.
  Vectorize = ((CurrentWgMethod == "loopvec" || CurrentWgMethod == "cbs") &&
               (Dev->spmd == CL_FALSE));
  PTO.SLPVectorization = Vectorize;
  PTO.LoopVectorization = Vectorize;
  OptimizeLevel = OLevel;
  SizeLevel = SLevel;

#ifdef DEBUG_NEW_PASS_MANAGER
  PrintPassOpts.Verbose = true;
  PrintPassOpts.SkipAnalyses = false;
  PrintPassOpts.Indent = true;
  SI.reset(new StandardInstrumentations(Context,
                                        true, // debug logging
                                        false, // verify each
                                        PrintPassOpts));
  SI->registerCallbacks(PIC, &MAM);
#endif
  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
#ifdef DEBUG_NEW_PASS_MANAGER
  PassB.reset(new PassBuilder(TM, PTO, std::nullopt, &PIC));
#else
  PassB.reset(new PassBuilder(TM, PTO));
#endif
  PassBuilder &PB = *PassB.get();

#if 0
  // TODO figure out why this doesn't work. Used to work with old PM,
  // but with the new PM, it still tries to use printf libcall
  // Register our TargetLibraryInfoImpl.
  TLII.reset(new TargetLibraryInfoImpl(DevTriple));
  // Disables automated generation of libcalls from code patterns.
  // TCE doesn't have a runtime linker which could link the libs later on.
  // Also the libcalls might be harmful for WG autovectorization where we
  // want to try to vectorize the code it converts to e.g. a memset or
  // a memcpy
  TLII->disableAllFunctions();
  // Analysis pass providing the \c TargetLibraryInfo:
  // TargetLibraryAnalysis

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

void PoclModulePassManager::run(llvm::Module &Bitcode) {
  PM.run(Bitcode, MAM);
#ifdef SEPARATE_OPTIMIZATION_FROM_POCL_PASSES
  populateModulePM(nullptr, (void *)&Bitcode, OptimizeLevel, SizeLevel,
                   Vectorize);
#endif
}

class TwoStagePoclModulePassManager {
  PoclModulePassManager Stage1;
  PoclModulePassManager Stage2;
#ifndef PER_STAGE_TARGET_MACHINE
  std::unique_ptr<llvm::TargetMachine> Machine;
#endif
public:
  TwoStagePoclModulePassManager() = default;
  llvm::Error build(cl_device_id Dev, const std::string &S1_Pipeline,
                    unsigned S1_OLevel, unsigned S1_SLevel,
                    const std::string &S2_Pipeline,
                    unsigned S2_OLevel, unsigned S2_SLevel);
  void run(llvm::Module &Bitcode);
};

llvm::Error TwoStagePoclModulePassManager::build(
    cl_device_id Dev, const std::string &S1_Pipeline, unsigned S1_OLevel,
    unsigned S1_SLevel, const std::string &S2_Pipeline, unsigned S2_OLevel,
    unsigned S2_SLevel) {

#ifndef PER_STAGE_TARGET_MACHINE
  Machine.reset(GetTargetMachine(Dev));
  TargetMachine *TMach = Machine.get();
#endif
  llvm::Error E1 = Stage1.build(S1_Pipeline,
                                S1_OLevel, S1_SLevel,
#ifndef PER_STAGE_TARGET_MACHINE
                                TMach,
#endif
                                Dev);
  if (E1)
    return E1;

  return Stage2.build(S2_Pipeline,
                      S2_OLevel, S2_SLevel,
#ifndef PER_STAGE_TARGET_MACHINE
                      TMach,
#endif
                      Dev);
}

void TwoStagePoclModulePassManager::run(llvm::Module &Bitcode) {
  Stage1.run(Bitcode);
  Stage2.run(Bitcode);
}

#endif

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
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
  Passes.push_back(PassName);
#else
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
    POCL_ABORT("unknown pass type");
  }
#endif
}

// for legacy PM, add each Analysis to the pass pipeline;
// for new PM, add them with proper nesting...  X(Y(Require<>))
static void addAnalysis(std::vector<std::string> &Passes, std::string PassName,
                        PassType T = PassType::Function) {
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
  Passes.push_back(PassName);
#else
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
    POCL_ABORT("unknown pass type");
  }
#endif
}

static void addStage1PassesToPipeline(cl_device_id Dev,
                                      std::vector<std::string> &Passes) {
  /* The kernel compiler passes to run, in order.

     Notes about the kernel compiler phase ordering:

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
  */

  // NOTE: if you add a new PoCL pass here,
  // don't forget to register it in registerPassBuilderPasses
  addPass(Passes, "fix-min-legal-vec-size", PassType::Module);
  addPass(Passes, "inline-kernels");
  addPass(Passes, "remove-optnone");
  addPass(Passes, "optimize-wi-func-calls");
  addPass(Passes, "handle-samplers");
  addPass(Passes, "infer-address-spaces");
  addAnalysis(Passes, "workitem-handler-chooser");
  addPass(Passes, "mem2reg");
  addAnalysis(Passes, "domtree");
  if (Dev->spmd != CL_FALSE) {
    addPass(Passes, "flatten-inline-all", PassType::Module);
    addPass(Passes, "always-inline", PassType::Module);
  } else {
    addPass(Passes, "flatten-globals", PassType::Module);
    addPass(Passes, "flatten-barrier-subs", PassType::Module);
    addPass(Passes, "always-inline", PassType::Module);
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
    addPass(Passes, "inline");
#endif
  }
  // this must be done AFTER inlining, see note above
  addPass(Passes, "automatic-locals", PassType::Module);

  // It should be now safe to run -O3 over the single work-item kernel
  // as the barrier has the attributes preventing illegal motions and
  // duplication. Let's do it to clean up the code for later Passes.
  // Especially the WI context structures get needlessly bloated in case there
  // is dead code lying around.
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
  addPass(Passes, "STANDARD_OPTS");
#else
  // the optimization for new PM is handled separately
#endif
}

static void addStage2PassesToPipeline(cl_device_id Dev,
                                      std::vector<std::string> &Passes) {

  // NOTE: if you add a new PoCL pass here,
  // don't forget to register it in registerPassBuilderPasses
  if (Dev->spmd == CL_FALSE) {
    addPass(Passes, "simplifycfg");
    addPass(Passes, "loop-simplify");

    // required for OLD PM
    addAnalysis(Passes, "workitem-handler-chooser");
    addAnalysis(Passes, "pocl-vua");
    addPass(Passes, "phistoallocas");
    addPass(Passes, "isolate-regions");

    // NEW PM requires WIH & VUA analyses here,
    // but they should not be invalidated by previous passes
    addPass(Passes, "implicit-loop-barriers", PassType::Loop);

    addPass(Passes, "implicit-cond-barriers");
    addPass(Passes, "loop-barriers", PassType::Loop);
    // required for new PM: WorkitemLoops to remove PHi nodes from LCSSA
    // 153: pocl::WorkitemLoopsImpl::addContextSaveRestore(llvm::Instruction*):
    // 153:   Assertion `"Cannot add context restore for a PHI node at the
    // region entry!" 153:   && RegionOfBlock( Phi->getParent())->entryBB() !=
    // Phi->getParent()' failed.
    addPass(Passes, "instcombine");

    addPass(Passes, "barriertails");
    addPass(Passes, "canon-barriers");
    addPass(Passes, "isolate-regions");

    // required for OLD PM
    addAnalysis(Passes, "wi-aa");
    addAnalysis(Passes, "workitem-handler-chooser");
    addAnalysis(Passes, "pocl-vua");

#if 0
    // use PoCL's own print-module pass
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
    addPass(Passes, "print-pocl-cfg");
#else
    // note: the "before" is an option given to the PoclCFGPrinter instance;
    // it will be used as a prefix to the dot files ("PREFIX_kernel.dot")
    addPass(Passes, "print<pocl-cfg;before>", PassType::Module);
#endif
#endif

    addPass(Passes, "workitemrepl");
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

  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running
  // the standard LLVM optimizations.
  addPass(Passes, "simplifycfg");

#if 0
  addPass(Passes, "print-module");
  addPass(Passes, "dot-cfg");
#endif

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
  addPass(Passes, "STANDARD_OPTS");

  // Due to unfortunate phase-ordering problems with store sinking,
  // loop deletion does not always apply when executing -O3 only
  // once. Cherry pick the optimization to rerun here.
  addPass(Passes, "loop-deletion");
  addPass(Passes, "remove-barriers");

#else
  // the optimization for new PM is handled separately
#endif
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
static bool runKernelCompilerPasses(cl_device_id device, llvm::Module &Mod) {

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  legacy::PassManager PM;

  // Need to setup the target info for target specific passes. */
  Triple triple(device->llvm_target_triplet);
  std::unique_ptr<llvm::TargetMachine> TM(GetTargetMachine(device));
  llvm::TargetMachine *Machine = TM.get();

  if (Machine)
    PM.add(
        createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));

  /* Disables automated generation of libcalls from code patterns.
     TCE doesn't have a runtime linker which could link the libs later on.
     Also the libcalls might be harmful for WG autovectorization where we
     want to try to vectorize the code it converts to e.g. a memset or
     a memcpy */
  TargetLibraryInfoImpl TLII(triple);
  TLII.disableAllFunctions();
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  std::vector<std::string> Passes;
  addStage1PassesToPipeline(device, Passes);
  addStage2PassesToPipeline(device, Passes);

  // Now actually add the listed passes to the PassManager.
  for (unsigned i = 0; i < Passes.size(); ++i) {
    // This is (more or less) -O3.
    if (Passes[i] == "STANDARD_OPTS") {
      // These need to be setup in addition to invoking the passes
      // to get the vectorizers initialized properly. Assume SPMD
      // devices do not want to vectorize intra work-item at this
      // stage.
      bool Vectorize =
          ((CurrentWgMethod == "loopvec" || CurrentWgMethod == "cbs") &&
           (device->spmd == CL_FALSE));
      populateModulePM(&PM, nullptr, 3, 0, Vectorize);
      continue;
    }

    const PassInfo *PIs = Registry->getPassInfo(StringRef(Passes[i]));
    if (PIs) {
      // std::cout << "-"<<Passes[i] << " ";
      Pass *thispass = PIs->createPass();
      PM.add(thispass);
    } else {
      std::cerr << "Failed to create kernel compiler pass " << Passes[i]
                << std::endl;
      return false;
    }
  }

  PM.run(Mod);
  return true;
}
#else

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

static bool runKernelCompilerPasses(cl_device_id device, llvm::Module &Mod) {

  // use new pass manager
  TwoStagePoclModulePassManager PM;
  std::vector<std::string> Passes1;
  addStage1PassesToPipeline(device, Passes1);
  std::string P1 = convertPassesToPipelineString(Passes1);
  std::vector<std::string> Passes2;
  addStage2PassesToPipeline(device, Passes2);
  std::string P2 = convertPassesToPipelineString(Passes2);

  Error E = PM.build(device, P1, 2, 0, P2, 3, 0);
  if (E) {
    std::cerr << "LLVM: failed to create compilation pipeline";
    return false;
  }

  PM.run(Mod);
  return true;
}
#endif

void pocl_destroy_llvm_module(void *modp, cl_context ctx) {

  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  llvm::Module *mod = (llvm::Module *)modp;
  if (mod) {
    delete mod;
    --llvm_ctx->number_of_IRs;
  }
}

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
                              LinkinOutputBCPath, 0);
    if (r != 0) {
      POCL_MSG_ERR("ProgramWithContext->init: failed to write module\n");
      return false;
    }

    if (pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT)) {
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

  bool getBitcodeForKernel(const char* KernelName,
                           char* OutputPath,
                           std::string *BuildLog) {
    std::lock_guard<std::mutex> LockGuard(Lock);

    // Create an empty Module and copy only the kernel+callgraph from
    // program.bc.
    std::unique_ptr<llvm::Module> KernelBC(
        new llvm::Module(llvm::StringRef("parallel_bc"), LLVMCtx));

    KernelBC->setTargetTriple(ProgramBC->getTargetTriple());
    KernelBC->setDataLayout(ProgramBC->getDataLayout());

    copyKernelFromBitcode(KernelName, KernelBC.get(), ProgramBC.get(), nullptr);

    if (pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT)) {
      llvm::raw_string_ostream Errs(*BuildLog);
      if (llvm::verifyModule(*KernelBC.get(), &Errs)) {
        POCL_MSG_ERR("Failed to verify Kernel Module:\n%s\n",
                     BuildLog->c_str());
        BuildLog->append("Failed to verify Kernel Module\n");
        return false;
      }
    }

    pocl_cache_tempname(OutputPath, ".bc", NULL);
    int r = pocl_write_module(KernelBC.get(), OutputPath, 0);
    if (r != 0) {
      POCL_MSG_ERR("getBitcodeForKernel: failed to write module\n");
      BuildLog->append("getBitcodeForKernel: failed to write module\n");
      return false;
    }
    return true;
  }
};

static int convertBitcodeToSpv(char* TempBitcodePath,
                               std::string *BuildLog,
                               char **SpirvContent,
                               uint64_t *SpirvSize) {

  char TempSpirvPath[POCL_MAX_PATHNAME_LENGTH];

// max bytes in output of 'llvm-spirv'
#define MAX_OUTPUT_BYTES 65536

//   --spirv-ext=<+SPV_extenstion1_name,-SPV_extension2_name>
//   Specify list of allowed/disallowed extensions
#define ALLOW_EXTS                                                             \
  "--spirv-ext=+SPV_INTEL_subgroups,+SPV_INTEL_usm_storage_classes,+SPV_"      \
  "INTEL_arbitrary_precision_integers,+SPV_INTEL_arbitrary_precision_fixed_"   \
  "point,+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_kernel_"     \
  "attributes"
  /*
  possibly useful:
    "+SPV_INTEL_unstructured_loop_controls,"
    "+SPV_INTEL_blocking_pipes,"
    "+SPV_INTEL_function_pointers,"
    "+SPV_INTEL_io_pipes,"
    "+SPV_INTEL_inline_assembly,"
    "+SPV_INTEL_optimization_hints,"
    "+SPV_INTEL_float_controls2,"
    "+SPV_INTEL_vector_compute,"
    "+SPV_INTEL_fast_composite,"
    "+SPV_INTEL_variable_length_array,"
    "+SPV_INTEL_fp_fast_math_mode,"
    "+SPV_INTEL_long_constant_composite,"
    "+SPV_INTEL_memory_access_aliasing,"
    "+SPV_INTEL_runtime_aligned,"
    "+SPV_INTEL_arithmetic_fence,"
    "+SPV_INTEL_bfloat16_conversion,"
    "+SPV_INTEL_global_variable_decorations,"
    "+SPV_INTEL_non_constant_addrspace_printf,"
    "+SPV_INTEL_hw_thread_queries,"
    "+SPV_INTEL_complex_float_mul_div,"
    "+SPV_INTEL_split_barrier,"
    "+SPV_INTEL_masked_gather_scatter"

  probably not useful:
    "+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,"
    "+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_fpga_memory_attributes,"
    "+SPV_INTEL_fpga_memory_accesses,"
    "+SPV_INTEL_fpga_reg,+SPV_INTEL_fpga_buffer_location,"
    "+SPV_INTEL_fpga_cluster_attributes,"
    "+SPV_INTEL_loop_fuse,"
    "+SPV_INTEL_optnone," // this one causes crash
    "+SPV_INTEL_fpga_dsp_control,"
    "+SPV_INTEL_fpga_invocation_pipelining_attributes,"
    "+SPV_INTEL_token_type,"
    "+SPV_INTEL_debug_module,"
    "+SPV_INTEL_joint_matrix,"
  */
  pocl_cache_tempname(TempSpirvPath, ".spirv", NULL);
  char LLVMspirv[] = LLVM_SPIRV;
  char AllowedExtOption[] = ALLOW_EXTS;
  // TODO ze_device_module_properties_t.spirvVersionSupported
  char MaxSPIRVOption[] = "--spirv-max-version=1.2";
#if (LLVM_MAJOR == 15) || (LLVM_MAJOR == 16)
#ifdef LLVM_OPAQUE_POINTERS
  char OpaquePtrsOption[] = "--opaque-pointers";
#endif
#endif
  char OutputOption[] = { '-', 'o', 0 };
  char *CmdArgs[] = { LLVMspirv, AllowedExtOption,
#if (LLVM_MAJOR == 15) || (LLVM_MAJOR == 16)
#ifdef LLVM_OPAQUE_POINTERS
                      OpaquePtrsOption,
#endif
#endif
                      MaxSPIRVOption, OutputOption,
                      TempSpirvPath, TempBitcodePath, NULL };
  char CapturedOutput[MAX_OUTPUT_BYTES];
  size_t CapturedBytes = MAX_OUTPUT_BYTES;

  int r =
      pocl_run_command_capture_output(CapturedOutput, &CapturedBytes, CmdArgs);
  if (r != 0) {
    BuildLog->append("llvm-spirv failed with output:\n");
    std::string Captured(CapturedOutput, CapturedBytes);
    BuildLog->append(Captured);
    return -1;
  }

  r = pocl_read_file(TempSpirvPath, SpirvContent, SpirvSize);
  if (r != 0) {
    BuildLog->append("failed to read output file from llvm-spirv\n");
    return -1;
  }

  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0) {
    pocl_remove(TempBitcodePath);
    pocl_remove(TempSpirvPath);
  } else {
    POCL_MSG_PRINT_GENERAL("LLVM SPIR-V conversion tempfiles: %s -> %s",
                           TempBitcodePath, TempSpirvPath);
  }
  return 0;
}

} // namespace pocl

void *pocl_llvm_create_context_for_program(const char *ProgramBcBytes,
                                           size_t ProgramBcSize,
                                           char **LinkinSpirvContent,
                                           uint64_t *LinkinSpirvSize) {
  assert(ProgramBcBytes);
  assert(ProgramBcSize > 0);

  char TempBitcodePath[POCL_MAX_PATHNAME_LENGTH];

  pocl::ProgramWithContext *P = new pocl::ProgramWithContext;
  // parse the program's bytes into a llvm::Module
  if (P == nullptr ||
      !P->init(ProgramBcBytes, ProgramBcSize, TempBitcodePath)) {
    POCL_MSG_ERR("failed to create program for context");
    return nullptr;
  }

  std::string BuildLog;
  if (pocl::convertBitcodeToSpv(TempBitcodePath, &BuildLog,
                                LinkinSpirvContent, LinkinSpirvSize) != 0) {
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
int pocl_llvm_extract_kernel_spirv(void* ProgCtx,
                                   const char* KernelName,
                                   void* BuildLogStr,
                                   char **SpirvContent,
                                   uint64_t *SpirvSize) {

  POCL_MEASURE_START(extractKernel);

  std::string *BuildLog = (std::string *)BuildLogStr;

  char TempBitcodePath[POCL_MAX_PATHNAME_LENGTH];
  pocl::ProgramWithContext *P = (pocl::ProgramWithContext *)ProgCtx;
  if (!P->getBitcodeForKernel(KernelName, TempBitcodePath, BuildLog)) {
    return -1;
  }

  int r = pocl::convertBitcodeToSpv(TempBitcodePath, BuildLog,
                                    SpirvContent, SpirvSize);

  POCL_MEASURE_FINISH(extractKernel);
  return r;
}

static int pocl_llvm_run_pocl_passes(llvm::Module *Bitcode,
                                     _cl_command_run *RunCommand, // optional
                                     llvm::LLVMContext *LLVMContext,
                                     PoclLLVMContextData *PoclCtx,
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

#ifdef DUMP_LLVM_PASS_TIMINGS
  llvm::TimePassesIsEnabled = true;
#endif
  POCL_MEASURE_START(llvm_workgroup_ir_func_gen);
  runKernelCompilerPasses(Device, *Bitcode);
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

  int res =
      pocl_llvm_run_pocl_passes(ParallelBC, RunCommand, LLVMContext,
                                PoCLLLVMContext, Kernel, Device, Specialize);

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
    Err = pocl_write_module((char *)ParallelBC, TempParallelBCFileName, 0);
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

static void initPassManagerForCodeGen(legacy::PassManager &PM,
                                      cl_device_id Device) {

  llvm::Triple Triple(Device->llvm_target_triplet);

  llvm::TargetLibraryInfoWrapperPass *TLIPass =
      new TargetLibraryInfoWrapperPass(Triple);
  PM.add(TLIPass);
}

/* Run LLVM codegen on input file (parallel-optimized).
 * modp = llvm::Module* of parallel.bc
 * Output native object file (<kernel>.so.o). */
int pocl_llvm_codegen(cl_device_id Device, cl_program program, void *Modp,
                      char **Output, uint64_t *OutputSize) {

  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  llvm::Module *Input = (llvm::Module *)Modp;
  assert(Input);
  *Output = nullptr;

  legacy::PassManager PMObj;
  initPassManagerForCodeGen(PMObj, Device);

  std::unique_ptr<llvm::TargetMachine> TM(GetTargetMachine(Device));
  llvm::TargetMachine *Target = TM.get();

  // First try direct object code generation from LLVM, if supported by the
  // LLVM backend for the target.
  bool LLVMGeneratesObjectFiles = true;

  SmallVector<char, 4096> Data;
  llvm::raw_svector_ostream SOS(Data);
  bool cannotEmitFile;

  cannotEmitFile = Target->addPassesToEmitFile(PMObj, SOS, nullptr,
                                  llvm::CGFT_ObjectFile);

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
  }

  legacy::PassManager PMAsm;
  initPassManagerForCodeGen(PMAsm, Device);

  POCL_MSG_PRINT_LLVM("Generating assembly text.\n");

  // The LLVM target does not implement support for emitting object file directly.
  // Have to emit the text first and then call the assembler from the command line
  // to produce the binary.

  if (Target->addPassesToEmitFile(PMAsm, SOS, nullptr,
                                  llvm::CGFT_AssemblyFile)) {
    POCL_ABORT("The target supports neither obj nor asm emission!");
  }




#ifdef DUMP_LLVM_PASS_TIMINGS
  llvm::TimePassesIsEnabled = true;
#endif
  // This produces the assembly text:
  PMAsm.run(*Input);
#ifdef DUMP_LLVM_PASS_TIMINGS
  llvm::reportAndResetTimings();
#endif

  // Next call the target's assembler via the Toolchain API indirectly through
  // the Driver API.

  char AsmFileName[POCL_MAX_PATHNAME_LENGTH];
  char ObjFileName[POCL_MAX_PATHNAME_LENGTH];

  std::string AsmStr = SOS.str().str();
  pocl_write_tempfile(AsmFileName, "/tmp/pocl-asm", ".s", AsmStr.c_str(),
                      AsmStr.size(), nullptr);
  pocl_mk_tempname(ObjFileName, "/tmp/pocl-obj", ".o", nullptr);

  const char *Args[] = {CLANG, AsmFileName, "-c", "-o", ObjFileName, nullptr};
  int Res = pocl_invoke_clang(Device, Args);

  if (Res == 0) {
    if (pocl_read_file(ObjFileName, Output, OutputSize))
      POCL_ABORT("Could not read the object file.");
  }

  pocl_remove(AsmFileName);
  pocl_remove(ObjFileName);
  return Res;

}

void populateModulePM(void *Passes, void *Module, unsigned OptL, unsigned SizeL,
                      bool Vectorize) {
#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
  PassManagerBuilder Builder;
  Builder.OptLevel = OptL;
  Builder.SizeLevel = SizeL;

  Builder.LoopVectorize = Vectorize;
  Builder.SLPVectorize = Vectorize;
  bool Verify =
      pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT);
  Builder.VerifyInput = Verify;
  Builder.VerifyOutput = Verify;

  llvm::legacy::PassManager *LegacyPasses = nullptr;
  llvm::legacy::PassManager PM;
  if (Passes) {
    LegacyPasses = (llvm::legacy::PassManager *)Passes;
  } else {
    LegacyPasses = &PM;
  }
  Builder.populateModulePassManager(*LegacyPasses);

  if (Module) {
    llvm::Module *Mod = (llvm::Module *)Module;
    LegacyPasses->run(*Mod);
  }
#else
  // Create the analysis managers.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  PassBuilder PB;

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
  ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(Opt);
  if (Module) {
    llvm::Module *Mod = (llvm::Module *)Module;
    MPM.run(*Mod, MAM);
  }
#endif
}

/* vim: set ts=4 expandtab: */
