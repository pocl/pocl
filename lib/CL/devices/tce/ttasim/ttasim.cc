/* ttasim.h - a pocl device driver for simulating TTA devices using TCE's ttasim

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "ttasim.h"
#include "bufalloc.h"
#include "pocl_device.h"
#include "pocl_util.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <string>

/* Supress some warnings because of including tce_config.h after pocl's config.h. */
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_VERSION
#undef VERSION
#undef SIZEOF_DOUBLE

#include <SimpleSimulatorFrontend.hh>
#include <Machine.hh>
#include <MemorySystem.hh>
#include <Program.hh>
#include <DataLabel.hh>
#include <SimulatorCLI.hh>
#include <SimulationEventHandler.hh>
#include <Listener.hh>

#include "tce_common.h"

using namespace TTAMachine;

size_t pocl_ttasim_max_work_item_sizes[] = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX};

static void *pocl_ttasim_thread (void *p);


class TTASimDevice : public TCEDevice {
public:
  TTASimDevice(cl_device_id dev, const char* adfName) :
    TCEDevice(dev, adfName), simulator(adfName), 
    simulatorCLI(simulator.frontend()), debuggerRequested(false),
    shutdownRequested(false) {
    char dev_name[256];
    if (device_count > 0)
      {
        if (snprintf (dev_name, 256, "ttasim%d", device_count) < 0)
          POCL_ABORT("Unable to generate the device name string.");
        dev->name = strdup(dev_name);  
      }
    ++device_count;

    SigINTHandler* ctrlcHandler = new SigINTHandler(this);
    Application::setSignalHandler(SIGINT, *ctrlcHandler);

    initMemoryManagement(simulator.machine());

    pthread_mutex_init (&lock, NULL);
    pthread_cond_init (&simulation_start_cond, NULL);
    pthread_create (&ttasim_thread, NULL, pocl_ttasim_thread, this);
  }

  ~TTASimDevice() {
    shutdownRequested = true;
    simulator.stop();
    pthread_cond_signal (&simulation_start_cond);
    pthread_join (ttasim_thread, NULL);
  }

  virtual void copyHostToDevice(const void *host_ptr, uint32_t dest_addr, size_t count) {
    MemorySystem &mems = simulator.memorySystem();
    MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
    for (std::size_t i = 0; i < count; ++i) {
      unsigned char val = ((char*)host_ptr)[i];
      globalMem->write (dest_addr + i, (Memory::MAU)(val));
    }
  }

  virtual void copyDeviceToHost(uint32_t src_addr, const void *host_ptr, 
                                size_t count) {
    /* Find the simulation model for the global address space. */
    MemorySystem &mems = simulator.memorySystem();
    MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
    for (std::size_t i = 0; i < count; ++i) {
      ((char*)host_ptr)[i] = globalMem->read (src_addr + i);
      //    printf("host: read byte %u from %d to %x\n",  ((char*)dst_ptr)[i], src_addr + i, &((char*)dst_ptr)[i]);
    }
  }

  virtual void loadProgramToDevice(const std::string& asmFile) {
    if (simulator.isRunning()) 
      simulator.stop();
    while (simulator.isRunning()) 
      ;
    /* Save the cycle count to maintain a global running cycle count
       over all the simulations. */
    if (currentProgram != NULL)
        globalCycleCount += simulator.cycleCount();
    simulator.loadProgram(asmFile);
    currentProgram = &simulator.program();
    initDataMemory();
  }

  virtual uint64_t timeStamp() {
    if (currentProgram == NULL) 
        return (uint64_t)(globalCycleCount * 1000.0f) / 
            parent->max_clock_frequency;

    return (uint64_t)(globalCycleCount + simulator.cycleCount()) * 1000.0f / 
                      parent->max_clock_frequency;
  }

  virtual void restartProgram() {
    pthread_cond_signal (&simulation_start_cond);
  }

  SimpleSimulatorFrontend simulator;
  /* A Command Line Interface for debugging. */
  SimulatorCLI simulatorCLI;
  volatile bool debuggerRequested;
  volatile bool shutdownRequested;

  pthread_t ttasim_thread;
  pthread_cond_t simulation_start_cond;
  pthread_mutex_t lock;

private:

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
    virtual void execute(int /*data*/, siginfo_t* /*info*/) {
      std::cerr << "### ctrl-C handler" << std::endl;
      d_->debuggerRequested = true;
      d_->simulator.stop();
      std::cerr << "### handler exit" << std::endl;
      /* Make it a one shot handler. Next Ctrl-C should kill the program. */
      Application::restoreSignalHandler(SIGINT);
    }
  private:
    TTASimDevice* d_;
  };  
  static int device_count;
};

int TTASimDevice::device_count = 0;

static 
void * 
pocl_ttasim_thread (void *p)
{
  TTASimDevice *d = (TTASimDevice*)p;
  do {
    if (d->shutdownRequested) pthread_exit(NULL);
    pthread_cond_wait (&d->simulation_start_cond, &d->lock);
    if (d->shutdownRequested) pthread_exit(NULL);
    do {
        d->simulator.run();
        if (d->debuggerRequested) {
            d->debuggerRequested = false;
            d->simulatorCLI.run();
            continue;
        }

        if (d->simulator.hadRuntimeError()) {
            d->simulatorCLI.run();
            POCL_ABORT("Runtime error in a ttasim device.");
        }
    } while (false);
  } while (true);
  return NULL;
}

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
    virtual void execute(int /*data*/, siginfo_t* /*info*/) {
        std::cerr << "### ctrl-C handler" << std::endl;
        d_->debuggerRequested = true;
        d_->simulator.stop();
        std::cerr << "### handler exit" << std::endl;
        /* Make it a one shot handler. Next Ctrl-C should kill the program. */
        Application::restoreSignalHandler(SIGINT);
    }
private:
  TTASimDevice* d_;
};


void
pocl_ttasim_init (cl_device_id device, const char* parameters)
{
  if (parameters == NULL)
    POCL_ABORT("The tta device requires the adf file as a device parameter.\n"
               "Set it with POCL_DEVICEn_PARAMETERS=\"path/to/themachine.adf\".\n");
  
  new TTASimDevice(device, parameters); 
}

void
pocl_ttasim_uninit (cl_device_id device)
{
  delete (TTASimDevice*)device->data;
}

void
pocl_ttasim_copy (void */*data*/, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb)
{
  POCL_ABORT_UNIMPLEMENTED();
  if (src_ptr == dst_ptr)
    return;
  
  memcpy (dst_ptr, src_ptr, cb);
}

void
pocl_ttasim_copy_rect (void */*data*/,
                      const void *__restrict const src_ptr,
                      void *__restrict__ const dst_ptr,
                      const size_t *__restrict__ const src_origin,
                      const size_t *__restrict__ const dst_origin, 
                      const size_t *__restrict__ const region,
                      size_t const src_row_pitch,
                      size_t const src_slice_pitch,
                      size_t const dst_row_pitch,
                      size_t const dst_slice_pitch)
{
  char const *__restrict const adjusted_src_ptr = 
    (char const*)src_ptr +
    src_origin[0] + src_row_pitch * (src_origin[1] + src_slice_pitch * src_origin[2]);
  char *__restrict__ const adjusted_dst_ptr = 
    (char*)dst_ptr +
    dst_origin[0] + dst_row_pitch * (dst_origin[1] + dst_slice_pitch * dst_origin[2]);
  
  size_t j, k;

  POCL_ABORT_UNIMPLEMENTED();

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
              adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
              region[0]);
}

void
pocl_ttasim_write_rect (void */*data*/,
                       const void *__restrict__ const host_ptr,
                       void *__restrict__ const device_ptr,
                       const size_t *__restrict__ const buffer_origin,
                       const size_t *__restrict__ const host_origin, 
                       const size_t *__restrict__ const region,
                       size_t const buffer_row_pitch,
                       size_t const buffer_slice_pitch,
                       size_t const host_row_pitch,
                       size_t const host_slice_pitch)
{
  char *__restrict const adjusted_device_ptr = 
    (char*)device_ptr +
    buffer_origin[0] + buffer_row_pitch * (buffer_origin[1] + buffer_slice_pitch * buffer_origin[2]);
  char const *__restrict__ const adjusted_host_ptr = 
    (char const*)host_ptr +
    host_origin[0] + host_row_pitch * (host_origin[1] + host_slice_pitch * host_origin[2]);
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  POCL_ABORT_UNIMPLEMENTED();
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              region[0]);
}

void
pocl_ttasim_read_rect (void */*data*/,
                      void *__restrict__ const host_ptr,
                      void *__restrict__ const device_ptr,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const host_origin, 
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      size_t const host_row_pitch,
                      size_t const host_slice_pitch)
{
  char const *__restrict const adjusted_device_ptr = 
    (char const*)device_ptr +
    buffer_origin[0] + buffer_row_pitch * (buffer_origin[1] + buffer_slice_pitch * buffer_origin[2]);
  char *__restrict__ const adjusted_host_ptr = 
    (char*)host_ptr +
    host_origin[0] + host_row_pitch * (host_origin[1] + host_slice_pitch * host_origin[2]);
  
  size_t j, k;
  
  /* TODO: handle overlaping regions */
  POCL_ABORT_UNIMPLEMENTED();
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              region[0]);
}

cl_ulong
pocl_ttasim_get_timer_value (void *data) 
{
  TTASimDevice *d = (TTASimDevice*)data;
  return d->timeStamp();
}
