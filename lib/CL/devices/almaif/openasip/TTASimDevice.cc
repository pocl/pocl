/* TTASimDevice.cc - basic way of accessing accelerator memory.
 *                 as a memory mapped region

   Copyright (c) 2022 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "TTASimDevice.hh"
#include "../AlmaifCompile.hh"
#include "../AlmaifShared.hh"
#include "TTASimControlRegion.hh"
#include "TTASimRegion.hh"

#include <AddressSpace.hh>
#include <Application.hh>
#include <Machine.hh>
#include <MemorySystem.hh>
#include <Procedure.hh>
#include <Program.hh>
#include <SimpleSimulatorFrontend.hh>
#include <SimulatorCLI.hh>

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "utlist.h"

#include <pthread.h>

#include <iostream>

static void *pocl_ttasim_thread(void *p);

/**
 * A handler class for Ctrl-c signal.
 *
 * Stops the simulation (if it's running) and attaches the ttasim
 * console to it.
 */
class SigINTHandler : public Application::UnixSignalHandler {
public:
  /**
   * Constructor.
   *
   * @param target The target SimulatorFrontend instance.
   */
  SigINTHandler(TTASimDevice *d) : d_(d) {}

  /**
   * Stops the simulation.
   */
  virtual void execute(int /*data*/, siginfo_t * /*info*/);

private:
  TTASimDevice *d_;
};

TTASimDevice::TTASimDevice(const std::string &adf_name) {

#ifdef ALMAIF_TTASimMMAP_DEBUG
  POCL_MSG_PRINT_ALMAIF("TTASimMMAP: Initializing TTASimMMAPregion with Address "
                       "%zu and Size %zu\n",
                       Address, RegionSize);
#endif
  std::string adf_char = adf_name + ".adf";

  simulator_ = new SimpleSimulatorFrontend(adf_char, false);
  assert(simulator_ != NULL && "simulator null\n");

  simulatorCLI_ = new SimulatorCLI(simulator_->frontend());

  SigINTHandler *ctrlcHandler = new SigINTHandler(this);
  Application::setSignalHandler(SIGINT, *ctrlcHandler);

  const TTAMachine::Machine &mach = simulator_->machine();

  ControlMemory = new TTASimControlRegion(mach, this);

  discoverDeviceParameters();

  TTAMachine::Machine::FunctionUnitNavigator funav =
      mach.functionUnitNavigator();

  TTAMachine::Machine::AddressSpaceNavigator nav = mach.addressSpaceNavigator();
  TTAMachine::AddressSpace *global_as = nullptr;
  TTAMachine::AddressSpace *cq_as = nullptr;

  for (int i = 0; i < nav.count(); ++i) {
    TTAMachine::AddressSpace *as = nav.item(i);
    if (as->hasNumericalId(TTA_ASID_GLOBAL)) {
      global_as = as;
    }
    if (as->hasNumericalId(TTA_ASID_CQ)) {
      cq_as = as;
    }
  }
  assert(global_as != nullptr);
  assert(cq_as != nullptr);

  MemorySystem &mems = simulator_->memorySystem();
  MemorySystem::MemoryPtr mem = mems.memory(*global_as);
  MemorySystem::MemoryPtr cq_mem = mems.memory(*cq_as);

  // Doesn't exist and should not ever be accessed
  InstructionMemory = nullptr;
  if ((global_as != cq_as) && !RelativeAddressing) {
    CQMemory = new TTASimRegion(0, CQSize, cq_mem);
  } else {
    CQMemory = new TTASimRegion(CQStart, CQSize, cq_mem);
  }
  DataMemory = new TTASimRegion(DmemStart, DmemSize, mem);

  // For built-in kernel use-case. If the firmware.tpef exists, load it in

  std::string tpef_file = adf_name + ".tpef";
  if (pocl_exists(tpef_file.c_str())) {
    POCL_MSG_PRINT_ALMAIF(
        "Almaif: Found built-in kernel firmware for ttasim. Loading it in.\n");
    loadProgram(tpef_file);
  } else {
    POCL_MSG_PRINT_ALMAIF(
        "File %s not found. Skipping program initialization\n",
        tpef_file.c_str());
  }

  if (!RelativeAddressing) {
    if (pocl_is_option_set("POCL_ALMAIF_EXTERNALREGION")) {
      char *region_params =
          strdup(pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "0,0"));
      char *save_ptr;
      char *param_token = strtok_r(region_params, ",", &save_ptr);
      size_t region_address = strtoul(param_token, NULL, 0);
      param_token = strtok_r(NULL, ",", &save_ptr);
      size_t region_size = strtoul(param_token, NULL, 0);
      if (region_size > 0) {
        memory_region_t *ext_region =
            (memory_region_t *)calloc(1, sizeof(memory_region_t));
        assert(ext_region && "calloc for ext memory_region_t failed");
        pocl_init_mem_region(ext_region, region_address, region_size);
        LL_APPEND(AllocRegions, ext_region);

        POCL_MSG_PRINT_ALMAIF(
            "Almaif: initialized external alloc region at %zx with size %zx\n",
            region_address, region_size);
        ExternalMemory = new TTASimRegion(region_address, region_size, mem);
      }
      free(region_params);
    }
  }

  POCL_INIT_LOCK(lock);
  POCL_INIT_COND(simulation_start_cond);
  pthread_create(&ttasim_thread, NULL, pocl_ttasim_thread, this);
}

TTASimDevice::~TTASimDevice() {
  shutdownRequested = true;
  simulator_->stop();
  POCL_LOCK(lock);
  POCL_SIGNAL_COND(simulation_start_cond);
  POCL_UNLOCK(lock);
  POCL_JOIN_THREAD(ttasim_thread);

  POCL_DESTROY_COND(simulation_start_cond);
  POCL_DESTROY_LOCK(lock);

  delete simulator_;
  delete simulatorCLI_;
}

void TTASimDevice::loadProgram(const std::string &tpef_file) {
  if (simulator_->isRunning())
    ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_RESET_CMD);
  while (simulator_->isRunning())
    ;
  /* Save the cycle count to maintain a global running cycle count
     over all the simulations. */
  // if (currentProgram != NULL)
  //  globalCycleCount += simulator_.cycleCount();
  simulator_->loadProgram(tpef_file.c_str());
}

void TTASimDevice::loadProgramToDevice(almaif_kernel_data_s *kd,
                                       cl_kernel kernel,
                                       _cl_command_node *cmd) {
  assert(kd);

  char tpef_file[POCL_MAX_PATHNAME_LENGTH];
  // first try specialized
  pocl_cache_kernel_cachedir_path(tpef_file, kernel->program,
                                  cmd->program_device_i, kernel,
                                  "/parallel.tpef", cmd, 1);
  if (!pocl_exists(tpef_file)) {
    // if it doesn't exist, try specialized with local sizes 0-0-0
    // should pick either 0-0-0 or 0-0-0-goffs0
    _cl_command_node cmd_copy;
    memcpy(&cmd_copy, cmd, sizeof(_cl_command_node));
    cmd_copy.command.run.pc.local_size[0] = 0;
    cmd_copy.command.run.pc.local_size[1] = 0;
    cmd_copy.command.run.pc.local_size[2] = 0;
    pocl_cache_kernel_cachedir_path(tpef_file, kernel->program,
                                    cmd->program_device_i, kernel,
                                    "/parallel.tpef", &cmd_copy, 1);
    if (!pocl_exists(tpef_file)) {
      pocl_cache_kernel_cachedir_path(tpef_file, kernel->program,
                                      cmd->program_device_i, kernel,
                                      "/parallel.tpef", &cmd_copy, 0);
    }
  }
  POCL_MSG_PRINT_ALMAIF("Loading kernel from file %s\n", tpef_file);

  loadProgram(tpef_file);

  std::string wg_func_name = std::string(kernel->name) + "_workgroup_argbuffer";
  const TTAProgram::Program *prog = &simulator_->program();
  if (prog->hasProcedure(wg_func_name)) {
    const TTAProgram::Procedure &proc = prog->procedure(wg_func_name);
    kd->kernel_address = proc.startAddress().location();
  } else {
    POCL_ABORT("Couldn't find wg_function procedure %s from the program\n",
               wg_func_name);
  }

  /*  for (int i=0; i<prog->procedureCount(); i++){
      POCL_MSG_PRINT_ALMAIF("procedurename=%s\n",
  prog->procedure(i).name().c_str());
  }*/
  // TODO Figure out kernel addres

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_CONTINUE_CMD);
  //  //currentProgram = &simulator_->program();
}

void *pocl_ttasim_thread(void *p) {
  TTASimDevice *d = (TTASimDevice *)p;
  do {
    if (d->shutdownRequested)
      goto EXIT;
    POCL_LOCK(d->lock);
    POCL_WAIT_COND(d->simulation_start_cond, d->lock);
    POCL_UNLOCK(d->lock);
    if (d->shutdownRequested)
      goto EXIT;

    d->simulator_->run();

    if (d->debuggerRequested) {
      d->debuggerRequested = false;
      d->simulatorCLI_->run();
      continue;
    }

    if (d->simulator_->hadRuntimeError()) {
      //            POCL_MSG_ERR ("Runtime error in a ttasim device! Launching
      //            debugger CLI \n"); d->simulatorCLI.run();
      POCL_ABORT("Runtime error in a ttasim device.\n");
    }

  } while (true);

EXIT:
  return NULL;
}

void TTASimDevice::restartProgram() {
  if (simulator_->isInitialized()) {
    POCL_LOCK(lock);
    POCL_SIGNAL_COND(simulation_start_cond);
    POCL_UNLOCK(lock);
  } else {
    POCL_MSG_PRINT_ALMAIF("Trying to start simulator without initialization\n");
  }
}

void TTASimDevice::stopProgram() { simulator_->stop(); }

void SigINTHandler::execute(int /*data*/, siginfo_t * /*info*/) {
  std::cerr << "### ctrl-C handler" << std::endl;
  d_->debuggerRequested = true;
  d_->simulator_->stop();
  std::cerr << "### handler exit" << std::endl;
  /* Make it a one shot handler. Next Ctrl-C should kill the program. */
  Application::restoreSignalHandler(SIGINT);
}
