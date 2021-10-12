
#include "TTASimDevice.h"
#include "TTASimRegion.h"
#include "TTASimControlRegion.h"
#include "accel-shared.h"

#include <SimpleSimulatorFrontend.hh>
#include <SimulatorCLI.hh>
#include <Machine.hh>
#include <MemorySystem.hh>
#include <AddressSpace.hh>
#include <Program.hh>
#include <Procedure.hh>
#include <Application.hh>

#include "pocl_cache.h"
#include "pocl_file_util.h"

#include <pthread.h>

#include <iostream>

static void *pocl_ttasim_thread (void *p);

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
    SigINTHandler(TTASimDevice* d) :
        d_(d) {
    }

    /**
     * Stops the simulation.
     */
    virtual void execute(int /*data*/, siginfo_t* /*info*/);

private:
  TTASimDevice* d_;
};


TTASimDevice::TTASimDevice(char *adf_name) {

#ifdef ACCEL_TTASimMMAP_DEBUG
  POCL_MSG_PRINT_INFO(
      "TTASimMMAP: Initializing TTASimMMAPregion with Address %zu and Size %zu\n",
      Address, RegionSize);
#endif
  char adf_char[120];
  snprintf(adf_char, sizeof(adf_char), "%s.adf", adf_name);


  simulator_ = new SimpleSimulatorFrontend(adf_char, false);
  assert(simulator_ != NULL && "simulator null\n");
  simulatorCLI_ = new SimulatorCLI(simulator_->frontend());


   SigINTHandler* ctrlcHandler = new SigINTHandler(this);
   Application::setSignalHandler(SIGINT, *ctrlcHandler);

  const TTAMachine::Machine& mach = simulator_->machine();
/*  try {
    mach = TTAMachine::Machine::loadFromADF(adf_char);
  } catch (Exception &e) {
    POCL_MSG_WARN("Error: %s\n",e.errorMessage().c_str());
    POCL_ABORT("Couldn't open mach\n");
  }
  mach->writeToADF("temp.adf");
*/
  ControlMemory = new TTASimControlRegion(mach, this);

  discoverDeviceParameters();
 
  TTAMachine::Machine::FunctionUnitNavigator funav = mach.functionUnitNavigator();

  TTAMachine::Machine::AddressSpaceNavigator nav = mach.addressSpaceNavigator();
  TTAMachine::AddressSpace *global_as = nullptr;
  for (int i = 0; i < nav.count(); ++i) {
    global_as = nav.item(i);
    if (global_as->hasNumericalId(TTA_ASID_GLOBAL)) {
      break;
    }
  }
  assert(global_as != nullptr);
  
  MemorySystem &mems = simulator_->memorySystem();
  MemorySystem::MemoryPtr mem = mems.memory(*global_as);


  //Doesn't exist and should not ever be accessed
  InstructionMemory = nullptr;
  CQMemory = new TTASimRegion(cq_start, cq_size, mem);
  DataMemory = new TTASimRegion(dmem_start, dmem_size, mem);


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

void
TTASimDevice::loadProgramToDevice(almaif_kernel_data_t *kd, cl_kernel kernel, _cl_command_node *cmd)
{
  assert(kd);

  char tpef_file[POCL_FILENAME_LENGTH];
  char cachedir[POCL_FILENAME_LENGTH];
  // first try specialized
  pocl_cache_kernel_cachedir_path(tpef_file, kernel->program, cmd->device_i,
      kernel, "/parallel.tpef", cmd, 1);
  if (!pocl_exists(tpef_file)) {
    // if it doesn't exist, try specialized with local sizes 0-0-0
    // should pick either 0-0-0 or 0-0-0-goffs0
    _cl_command_node cmd_copy;
    memcpy(&cmd_copy, cmd, sizeof(_cl_command_node));
    cmd_copy.command.run.pc.local_size[0] = 0;
    cmd_copy.command.run.pc.local_size[1] = 0;
    cmd_copy.command.run.pc.local_size[2] = 0;
    pocl_cache_kernel_cachedir_path(tpef_file, kernel->program, cmd->device_i,
        kernel, "/parallel.tpef", &cmd_copy, 1);
    POCL_MSG_PRINT_INFO("Specialized kernel not found, using %s\n", cachedir);
  }

  char wg_func_name[120];
  snprintf(wg_func_name, sizeof(wg_func_name), "%s_workgroup_argbuffer", kernel->name);
  const TTAMachine::Machine& mach = simulator_->machine();
  const TTAProgram::Program* prog = TTAProgram::Program::loadFromTPEF(tpef_file, mach);
  if (prog->hasProcedure(wg_func_name)){
    const TTAProgram::Procedure& proc = prog->procedure(wg_func_name);
    kd->kernel_address = proc.startAddress().location();
  } else {
      POCL_ABORT("Couldn't find wg_function procedure %s from the program\n", wg_func_name);
  }
  
  /*  for (int i=0; i<prog->procedureCount(); i++){
      POCL_MSG_PRINT_INFO("procedurename=%s\n", prog->procedure(i).name().c_str());
  }*/
  //TODO Figure out kernel addres



  if (simulator_->isRunning()) 
    ControlMemory->Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_RESET_CMD);
  while (simulator_->isRunning()) 
    ;
  /* Save the cycle count to maintain a global running cycle count
     over all the simulations. */
  //if (currentProgram != NULL)
  //  globalCycleCount += simulator_.cycleCount();

  simulator_->loadProgram(tpef_file);
  // Initialize AQL queue by setting all headers to invalid

  ControlMemory->Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_CONTINUE_CMD);
//  //currentProgram = &simulator_->program();
}

  void * 
pocl_ttasim_thread (void *p)
{
  TTASimDevice *d = (TTASimDevice*)p;
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
  if(simulator_->isInitialized()){
    POCL_LOCK(lock);
    POCL_SIGNAL_COND(simulation_start_cond);
    POCL_UNLOCK(lock);
  }
  else {
    POCL_MSG_PRINT_INFO("Trying to start simulator without initialization\n");
  }
}




void SigINTHandler::execute(int /*data*/, siginfo_t* /*info*/) {
    std::cerr << "### ctrl-C handler" << std::endl;
    d_->debuggerRequested = true;
    d_->simulator_->stop();
    std::cerr << "### handler exit" << std::endl;
    /* Make it a one shot handler. Next Ctrl-C should kill the program. */
    Application::restoreSignalHandler(SIGINT);
}

