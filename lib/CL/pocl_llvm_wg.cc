/* pocl_llvm_wg.cc: part of pocl LLVM API dealing with parallel.bc,
   optimization passes and codegen.

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

#include "config.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_llvm_api.h"
#include "pocl_file_util.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/Support/Casting.h>
#include <llvm/Support/MutexGuard.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/CommandLine.h>

#include <llvm/ADT/Triple.h>
#include <llvm/ADT/StringRef.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>

#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>

#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/PassRegistry.h>
#include <llvm/PassInfo.h>

#ifdef LLVM_OLDER_THAN_3_7
#include <llvm/PassManager.h>
#include <llvm/Target/TargetLibraryInfo.h>
#else
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#define PassManager legacy::PassManager
#endif

#include "linker.h"

using namespace llvm;

/**
 * Prepare the kernel compiler passes.
 *
 * The passes are created only once per program run per device.
 * The returned pass manager should not be modified, only the Module
 * should be optimized using it.
 */

static std::map<cl_device_id, llvm::TargetMachine *> targetMachines;
static std::map<cl_device_id, PassManager *> kernelPasses;

// This is used to control the kernel we process in the kernel compilation.
extern cl::opt<std::string> KernelName;

/* FIXME: these options should come from the cl_device, and
 * cl_program's options. */
static llvm::TargetOptions GetTargetOptions() {
  llvm::TargetOptions Options;
#ifdef LLVM_OLDER_THAN_3_9
  Options.PositionIndependentExecutable = true;
#endif
#ifdef HOST_FLOAT_SOFT_ABI
  Options.FloatABIType = FloatABI::Soft;
#else
  Options.FloatABIType = FloatABI::Hard;
#endif
  return Options;
}

void clearTargetMachines() {
  for (auto i = targetMachines.begin(), e = targetMachines.end(); i != e; ++i) {
    delete (llvm::TargetMachine *)i->second;
  }
  targetMachines.clear();
}

void clearKernelPasses() {
  for (auto i = kernelPasses.begin(), e = kernelPasses.end();
       i != e; ++i) {
    PassManager *pm = (PassManager *)i->second;
    delete pm;
  }

  kernelPasses.clear();
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine *GetTargetMachine(cl_device_id device, Triple &triple) {

  if (targetMachines.find(device) != targetMachines.end())
    return targetMachines[device];

  std::string Error;
  // Triple TheTriple(device->llvm_target_triplet);

  std::string MCPU = device->llvm_cpu ? device->llvm_cpu : "";

  const Target *TheTarget = TargetRegistry::lookupTarget("", triple, Error);

  // In LLVM 3.4 and earlier, the target registry falls back to
  // the cpp backend in case a proper match was not found. In
  // that case simply do not use target info in the compilation
  // because it can be an off-tree target not registered at
  // this point (read: TCE).
  if (!TheTarget || TheTarget->getName() == std::string("cpp")) {
    return 0;
  }

#ifdef LLVM_OLDER_THAN_6_0
  TargetMachine *TM = TheTarget->createTargetMachine(
      triple.getTriple(), MCPU, StringRef(""), GetTargetOptions(), Reloc::PIC_,
      CodeModel::Default, CodeGenOpt::Aggressive);
#else
  TargetMachine *TM = TheTarget->createTargetMachine(
      triple.getTriple(), MCPU, StringRef(""), GetTargetOptions(), Reloc::PIC_,
      CodeModel::Small, CodeGenOpt::Aggressive);
#endif

  assert(TM != NULL && "llvm target has no targetMachine constructor");
  if (device->ops->init_target_machine)
    device->ops->init_target_machine(device->data, TM);
  targetMachines[device] = TM;

  return TM;
}
/* helpers copied from LLVM opt END */

static PassManager &
kernel_compiler_passes(cl_device_id device, llvm::Module *input,
                       const std::string &module_data_layout) {

  PassManager *Passes = nullptr;
  PassRegistry *Registry = nullptr;

  if (kernelPasses.find(device) != kernelPasses.end()) {
    return *kernelPasses[device];
  }

  bool SPMDDevice = device->spmd;

  Registry = PassRegistry::getPassRegistry();

  Passes = new PassManager();

  // Need to setup the target info for target specific passes. */
  Triple triple(device->llvm_target_triplet);
  TargetMachine *Machine = GetTargetMachine(device, triple);

#ifdef LLVM_OLDER_THAN_3_7
  // Add internal analysis passes from the target machine.
  if (Machine)
    Machine->addAnalysisPasses(*Passes);
#else
  if (Machine)
    Passes->add(
        createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));
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

     -implicit-cond-barriers after -implicit-loop-barriers because the latter
     can inject barriers to loops inside conditional regions after which the
     peeling should be avoided by injecting the implicit conditional barriers.

     -loop-barriers, -barriertails, and -barriers should be ran after the
     implicit barrier injection passes so they "normalize" the implicit
     barriers also.

     -phistoallocas before -workitemloops as otherwise it cannot inject context
     restore code (PHIs need to be at the beginning of the BB and so one cannot
     context restore them with non-PHI code if the value is needed in another
     PHI). */

  std::vector<std::string> passes;
  passes.push_back("remove-optnone");
  passes.push_back("optimize-wi-func-calls");
  passes.push_back("handle-samplers");
  passes.push_back("workitem-handler-chooser");
  passes.push_back("mem2reg");
  passes.push_back("domtree");
  if (device->autolocals_to_args)
    passes.push_back("automatic-locals");

  if (SPMDDevice) {
    passes.push_back("flatten-inline-all");
    passes.push_back("always-inline");
  } else {
    passes.push_back("flatten-globals");
    passes.push_back("flatten-barrier-subs");
    passes.push_back("always-inline");
#ifndef LLVM_3_9
    passes.push_back("inline");
#endif
  }

#ifndef LLVM_OLDER_THAN_4_0
  // It should be now safe to run -O3 over the single work-item kernel
  // as the barrier has the attributes preventing illegal motions and
  // duplication. Let's do it to clean up the code for later passes.
  // Especially the WI context structures get needlessly bloated in case there
  // is dead code lying around.
  passes.push_back("STANDARD_OPTS");
#else
  // Just clean up any unused globals.
  passes.push_back("globaldce");
#endif

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
    // Remove the (pseudo) barriers.   They have no use anymore due to the
    // work-item loop control taking care of them.
    passes.push_back("remove-barriers");
  }

  // Add the work group launcher functions and privatize the pseudo variable
  // (local id) accesses. We have to do this late because we rely on aggressive
  // inlining to expose the _{local,group}_id accesses which will be replaced
  // with context struct accesses. TODO: A cleaner and a more robust way would
  // be to add hidden context struct parameters to the builtins that need the
  // context data and fix the calls early.
  if (device->workgroup_pass) {
    passes.push_back("workgroup");
    passes.push_back("always-inline");
  }

  // Attempt to move all allocas to the entry block to avoid the need for
  // dynamic stack which is problematic for some architectures.
  passes.push_back("allocastoentry");

#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  // Convert the semantical OpenCL address space IDs to the ones of the target.
  passes.push_back("target-address-spaces");
#endif

  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running
  // the standard LLVM optimizations.
  passes.push_back("simplifycfg");

#if 0
  passes.push_back("print-module");
  passes.push_back("dot-cfg");
#endif

  if (currentWgMethod == "loopvec" && SPMDDevice)
    passes.push_back("scalarizer");

  passes.push_back("instcombine");
  passes.push_back("STANDARD_OPTS");
  passes.push_back("instcombine");

  // Due to unfortunate phase-ordering problems with store sinking,
  // loop deletion does not always apply when executing -O3 only
  // once. Cherry pick the optimization to rerun here.
  passes.push_back("loop-deletion");

  // Now actually add the listed passes to the PassManager.
  for (unsigned i = 0; i < passes.size(); ++i) {
    // This is (more or less) -O3.
    if (passes[i] == "STANDARD_OPTS") {
      PassManagerBuilder Builder;
      Builder.OptLevel = 3;
      Builder.SizeLevel = 0;

      // These need to be setup in addition to invoking the passes
      // to get the vectorizers initialized properly.
      if (currentWgMethod == "loopvec") {
        Builder.LoopVectorize = true;
        Builder.SLPVectorize = true;
      }
#ifndef POCL_USE_FAKE_ADDR_SPACE_IDS
      Builder.VerifyInput = true;
      Builder.VerifyOutput = true;
#endif
      Builder.populateModulePassManager(*Passes);
      continue;
    }

    const PassInfo *PIs = Registry->getPassInfo(StringRef(passes[i]));
    if (PIs) {
      // std::cout << "-"<<passes[i] << " ";
      Pass *thispass = PIs->createPass();
      Passes->add(thispass);
    } else {
      std::cerr << "Failed to create kernel compiler pass " << passes[i]
                << std::endl;
      POCL_ABORT("FAIL\n");
    }
  }

  kernelPasses[device] = Passes;
  return *Passes;
}

void pocl_destroy_llvm_module(void *modp) {

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  llvm::Module *mod = (llvm::Module *)modp;
  if (mod) {
    delete mod;
    --numberOfIRs;
  }
}

// The global variables used to control the WG function generation's
// specialization parameteres. Defined in lib/llvmopencl/WorkitemHandler.cc.
namespace pocl {
extern size_t WGLocalSizeX;
extern size_t WGLocalSizeY;
extern size_t WGLocalSizeZ;
extern bool WGDynamicLocalSize;
extern size_t WGMaxGridDimWidth;
extern bool WGAssumeZeroGlobalOffset;
}

int pocl_update_program_llvm_irs_unlocked(cl_program program,
                                          unsigned device_i);

int pocl_llvm_generate_workgroup_function_nowrite(
    unsigned DeviceI, cl_device_id Device, cl_kernel Kernel,
    _cl_command_node *Command, void **Output, int Specialize) {

  _cl_command_run *RunCommand = &Command->command.run;
  cl_program Program = Kernel->program;

  currentPoclDevice = Device;

  PoclCompilerMutexGuard LockHolder(NULL);
  InitializeLLVM();

  if (Program->llvm_irs[DeviceI] == NULL)
    pocl_update_program_llvm_irs_unlocked(Program, DeviceI);

  llvm::Module *ProgramBC = (llvm::Module *)Program->llvm_irs[DeviceI];

  // Create an empty Module and copy only the kernel+callgraph from
  // program.bc.
  llvm::Module *ParallelBC =
      new llvm::Module(StringRef("parallel_bc"), GlobalContext());

  ParallelBC->setTargetTriple(ProgramBC->getTargetTriple());
  ParallelBC->setDataLayout(ProgramBC->getDataLayout());

  copyKernelFromBitcode(Kernel->name, ParallelBC, ProgramBC);

  // Set the specialization properties.
  if (Specialize) {
    pocl::WGLocalSizeX = RunCommand->pc.local_size[0];
    pocl::WGLocalSizeY = RunCommand->pc.local_size[1];
    pocl::WGLocalSizeZ = RunCommand->pc.local_size[2];
    pocl::WGDynamicLocalSize = pocl::WGLocalSizeX == 0 &&
                               pocl::WGLocalSizeY == 0 &&
                               pocl::WGLocalSizeZ == 0;
    pocl::WGAssumeZeroGlobalOffset = RunCommand->pc.global_offset[0] == 0 &&
                                     RunCommand->pc.global_offset[1] == 0 &&
                                     RunCommand->pc.global_offset[2] == 0;
    // Compile a smallgrid version or a generic one?
    if (RunCommand->force_large_grid_wg_func ||
        pocl_cmd_max_grid_dim_width(RunCommand) >=
            Device->grid_width_specialization_limit) {
      pocl::WGMaxGridDimWidth = 0; // The generic / large / unlimited size one.
    } else {
      // Limited grid dimension width by the device specific limit.
      pocl::WGMaxGridDimWidth = Device->grid_width_specialization_limit;
    }
  } else {
    pocl::WGDynamicLocalSize = true;
    pocl::WGLocalSizeX = pocl::WGLocalSizeY = pocl::WGLocalSizeZ = 0;
    pocl::WGAssumeZeroGlobalOffset = false;
    pocl::WGMaxGridDimWidth = 0;
  }

  KernelName = Kernel->name;

#ifdef LLVM_OLDER_THAN_3_7
  kernel_compiler_passes(Device, ParallelBC,
                         ParallelBC->getDataLayout()->getStringRepresentation())
      .run(*ParallelBC);
#else
  kernel_compiler_passes(Device, ParallelBC,
                         ParallelBC->getDataLayout().getStringRepresentation())
      .run(*ParallelBC);
#endif

  assert(Output != NULL);
  *Output = (void *)ParallelBC;
  ++numberOfIRs;
  return 0;
}

int pocl_llvm_generate_workgroup_function(unsigned DeviceI, cl_device_id Device,
                                          cl_kernel Kernel,
                                          _cl_command_node *Command,
                                          int Specialize) {

  void *Module = NULL;

  char ParallelBCPath[POCL_FILENAME_LENGTH];
  pocl_cache_work_group_function_path(ParallelBCPath, Kernel->program, DeviceI,
                                      Kernel, Command, Specialize);

  if (pocl_exists(ParallelBCPath))
    return CL_SUCCESS;

  char FinalBinaryPath[POCL_FILENAME_LENGTH];
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

  pocl_destroy_llvm_module(Module);
  return Error;
}

int pocl_update_program_llvm_irs_unlocked(cl_program program,
                                          unsigned device_i) {

  char program_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_program_bc_path(program_bc_path, program, device_i);

  if (!pocl_exists(program_bc_path))
    {
      POCL_MSG_ERR ("%s does not exist!\n",
                     program_bc_path);
      return -1;
    }

  assert(program->llvm_irs[device_i] == nullptr);
  program->llvm_irs[device_i] = parseModuleIR(program_bc_path);
  ++numberOfIRs;
  return 0;
}

int pocl_update_program_llvm_irs(cl_program program,
                                 unsigned device_i) {
  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  return pocl_update_program_llvm_irs_unlocked(program, device_i);
}

void pocl_free_llvm_irs(cl_program program, unsigned device_i) {
  if (program->llvm_irs[device_i]) {
    PoclCompilerMutexGuard lockHolder(NULL);
    InitializeLLVM();
    llvm::Module *mod = (llvm::Module *)program->llvm_irs[device_i];
    delete mod;
    --numberOfIRs;
    program->llvm_irs[device_i] = NULL;
  }
}

void pocl_llvm_update_binaries(cl_program program) {

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  char program_bc_path[POCL_FILENAME_LENGTH];
  int error;

  // Dump the LLVM IR Modules to memory buffers.
  assert(program->llvm_irs != NULL);
#ifdef DEBUG_POCL_LLVM_API
  printf("### refreshing the binaries of the program %p\n", program);
#endif

  for (size_t i = 0; i < program->num_devices; ++i) {
    assert(program->llvm_irs[i] != NULL);
    if (program->binaries[i])
      continue;

    pocl_cache_program_bc_path(program_bc_path, program, i);
    error = pocl_write_module((llvm::Module *)program->llvm_irs[i],
                              program_bc_path, 1);
    assert(error == 0);
    if (error)
      {
        POCL_MSG_ERR ("pocl_write_module(%s) failed!\n",
                     program_bc_path);
        continue;
      }

    std::string content;
    writeModuleIR((llvm::Module *)program->llvm_irs[i], content);

    size_t n = content.size();
    if (n < program->binary_sizes[i])
      POCL_ABORT("binary size doesn't match the expected value\n");
    if (program->binaries[i])
      POCL_MEM_FREE(program->binaries[i]);
    program->binaries[i] = (unsigned char *)malloc(n);
    std::memcpy(program->binaries[i], content.c_str(), n);

#ifdef DEBUG_POCL_LLVM_API
    printf("### binary for device %zi was of size %zu\n", i,
           program->binary_sizes[i]);
#endif
  }
}

static void initPassManagerForCodeGen(PassManager& PM, cl_device_id Device) {

  llvm::Triple Triple(Device->llvm_target_triplet);

  llvm::TargetLibraryInfoWrapperPass *TLIPass =
      new TargetLibraryInfoWrapperPass(Triple);
  PM.add(TLIPass);
}

/* Run LLVM codegen on input file (parallel-optimized).
 * modp = llvm::Module* of parallel.bc
 * Output native object file (<kernel>.so.o). */
int pocl_llvm_codegen(cl_device_id Device, void *Modp, char **Output,
                      uint64_t *OutputSize) {

  PoclCompilerMutexGuard LockHolder(nullptr);
  InitializeLLVM();

  llvm::Module *Input = (llvm::Module *)Modp;
  assert(Input);
  *Output = nullptr;

  PassManager PMObj;
  initPassManagerForCodeGen(PMObj, Device);

  llvm::Triple Triple(Device->llvm_target_triplet);
  llvm::TargetMachine *Target = GetTargetMachine(Device, Triple);

  // First try direct object code generation from LLVM, if supported by the
  // LLVM backend for the target.
  bool LLVMGeneratesObjectFiles = true;

  SmallVector<char, 4096> Data;
  llvm::raw_svector_ostream SOS(Data);
  bool cannotEmitFile;

#ifdef LLVM_OLDER_THAN_7_0
  cannotEmitFile = Target->addPassesToEmitFile(PMObj, SOS,
                                  TargetMachine::CGFT_ObjectFile);
#else
  cannotEmitFile = Target->addPassesToEmitFile(PMObj, SOS, nullptr,
                                  TargetMachine::CGFT_ObjectFile);
#endif

#ifdef LLVM_OLDER_THAN_5_0
  LLVMGeneratesObjectFiles = true;
#else
  LLVMGeneratesObjectFiles = !cannotEmitFile;
#endif

  if (LLVMGeneratesObjectFiles) {
    POCL_MSG_PRINT_LLVM("Generating an object file directly.\n");
    PMObj.run(*Input);
    std::string O = SOS.str(); // flush
    const char *Cstr = O.c_str();
    size_t S = O.size();
    *Output = (char *)malloc(S);
    *OutputSize = S;
    memcpy(*Output, Cstr, S);
    return 0;
  }

#ifdef LLVM_OLDER_THAN_5_0
  return 0;
#else

  PassManager PMAsm;
  initPassManagerForCodeGen(PMAsm, Device);

  POCL_MSG_PRINT_LLVM("Generating assembly text.\n");

  // The LLVM target does not implement support for emitting object file directly.
  // Have to emit the text first and then call the assembler from the command line
  // to produce the binary.
#ifdef LLVM_OLDER_THAN_3_7
  POCL_ABORT("Assembly text output support not implemented for LLVM < 3.7.");
#else
#ifdef LLVM_OLDER_THAN_7_0
  if (Target->addPassesToEmitFile(PMAsm, SOS,
                                  TargetMachine::CGFT_AssemblyFile)) {
    POCL_ABORT("The target supports neither obj nor asm emission!");
  }
#else
  if (Target->addPassesToEmitFile(PMAsm, SOS, nullptr,
                                  TargetMachine::CGFT_AssemblyFile)) {
    POCL_ABORT("The target supports neither obj nor asm emission!");
  }
#endif
#endif

  // This produces the assembly text:
  PMAsm.run(*Input);

  // Next call the target's assembler via the Toolchain API indirectly through
  // the Driver API.

  char AsmFileName[POCL_FILENAME_LENGTH];
  char ObjFileName[POCL_FILENAME_LENGTH];

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

#endif
}
/* vim: set ts=4 expandtab: */
