/* ttasim.cc - a pocl device driver for simulating TTA devices using TCE's ttasim

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
#include "install-paths.h"
#include "bufalloc.h"
#include "pocl_device.h"
#include "pocl_util.h"
#include "common.h"

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
#include <TCEString.hh>

#include <fstream>

#include "tce_common.h"

using namespace TTAMachine;

size_t pocl_ttasim_max_work_item_sizes[] = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX};

static void *pocl_ttasim_thread (void *p);

void
pocl_ttasim_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "ttasim";

  ops->probe = pocl_ttasim_probe;
  ops->init_device_infos = pocl_ttasim_init_device_infos;
  ops->uninit = pocl_ttasim_uninit;
  ops->init = pocl_ttasim_init;
  ops->malloc = pocl_tce_malloc;
  ops->alloc_mem_obj = pocl_tce_alloc_mem_obj;
  ops->create_sub_buffer = pocl_tce_create_sub_buffer;
  ops->free = pocl_tce_free;
  ops->read = pocl_tce_read;
  ops->read_rect = pocl_tce_read_rect;
  ops->write = pocl_tce_write;
  ops->write_rect = pocl_tce_write_rect;
  ops->copy = pocl_tce_copy;
  ops->copy_rect = pocl_tce_copy_rect;
  ops->map_mem = pocl_tce_map_mem;
  ops->run = pocl_tce_run;
  ops->get_timer_value = pocl_ttasim_get_timer_value;
  ops->init_build = pocl_tce_init_build;
}


void
pocl_ttasim_init_device_infos(struct _cl_device_id* dev)
{
  dev->type = CL_DEVICE_TYPE_GPU;
  dev->max_compute_units = 1;
  dev->max_work_item_dimensions = 3;
  dev->max_work_item_sizes[0] = CL_INT_MAX;
  dev->max_work_item_sizes[1] = CL_INT_MAX;
  dev->max_work_item_sizes[2] = CL_INT_MAX;
  dev->max_work_group_size = 8192;
  dev->preferred_wg_size_multiple = 8;
  dev->preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  /* TODO: figure out what the difference between preferred and native widths are. */
  dev->native_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->native_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->native_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->native_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->native_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->native_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->native_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  dev->max_clock_frequency = 100;
  dev->image_support = CL_FALSE;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
  dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
  dev->global_mem_cache_type = CL_NONE;
  dev->local_mem_type = CL_GLOBAL;
  dev->error_correction_support = CL_FALSE;
  dev->host_unified_memory = CL_FALSE;
  dev->endian_little = CL_FALSE;
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->queue_properties = CL_QUEUE_PROFILING_ENABLE;
  dev->vendor = "TTA-Based Co-design Environment";
  dev->profile = "EMBEDDED_PROFILE";
  dev->extensions = "";
  dev->llvm_target_triplet = "tce-tut-llvm";
  dev->has_64bit_long = 1;
}

unsigned int
pocl_ttasim_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  if(env_count < 0)
    return 0;

  return env_count;
}


class TTASimDevice : public TCEDevice {
public:
  TTASimDevice(cl_device_id dev, const char* adfName) :
    TCEDevice(dev, adfName), simulator(adfName, false), 
    simulatorCLI(simulator.frontend()), debuggerRequested(false),
    shutdownRequested(false), produceStandAloneProgram_(true) {
    char dev_name[256];

    const char *adf = strrchr(adfName, '/');
    if (adf != NULL && *adf != NULL) adf++;
    if (snprintf (dev_name, 256, "ttasim-%s", adf) < 0)
      POCL_ABORT("Unable to generate the device name string.");
    dev->long_name = strdup(dev_name);  
    ++device_count;

    SigINTHandler* ctrlcHandler = new SigINTHandler(this);
    Application::setSignalHandler(SIGINT, *ctrlcHandler);

    setMachine(simulator.machine());

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

  virtual void notifyKernelRunCommandSent
  (__kernel_exec_cmd& dev_cmd, _cl_command_run *run_cmd) {
    if (!produceStandAloneProgram_) return;

    static int runCounter = 0;
    TCEString tempDir = run_cmd->tmp_dir;
    TCEString baseFname = tempDir + "/";
    baseFname << "standalone_" << runCounter;

    TCEString buildScriptFname = baseFname + "_build";
    TCEString fname = baseFname + ".c";

    std::ofstream out(fname.c_str());

    out << "#include <lwpr.h>" << std::endl;
    out << "#include <pocl_device.h>" << std::endl << std::endl;

    out << "#define __local__ __attribute__((address_space(0)))" << std::endl;
    out << "#define __global__ __attribute__((address_space(3)))" << std::endl;
    out << "#define __constant__ __attribute__((address_space(3)))" << std::endl << std::endl;
    out << "typedef volatile __global__ __kernel_exec_cmd kernel_exec_cmd;" << std::endl;

    /* Need to byteswap back as we are writing C code. */
#define BSWAP(__X) byteswap_uint32_t (__X, needsByteSwap)

    /* The standalone binary shall have the same input data as in the original
       kernel host-device kernel launch command. The data is put into initialized 
       global arrays to easily exclude the initialization time from the execution
       time. Otherwise, the same command data is used for reproducing the execution.
       For example, the local memory allocations (addresses) are the same as in
       the original one. */

    /* Create the global buffers along with their initialization data. */
    for (int i = 0; i < run_cmd->kernel->num_args; ++i)
      {
        struct pocl_argument *al = &(run_cmd->arguments[i]);
        if (run_cmd->kernel->arg_is_pointer[i])
          {
            if (al->value == NULL) continue;
            unsigned start_addr = 
              ((chunk_info_t*)((*(cl_mem *) (al->value))->device_ptrs[parent->dev_id].mem_ptr))->start_address;
            unsigned size = 
              ((chunk_info_t*)((*(cl_mem *) (al->value))->device_ptrs[parent->dev_id].mem_ptr))->size;

            out << "__global__ char buffer_" << std::hex << start_addr 
                << "[] = {" << std::endl << "\t";

            MemorySystem &mems = simulator.memorySystem();
            MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
            for (std::size_t c = 0; c < size; ++c) {
              unsigned char val = globalMem->read (start_addr + c);
              out << "0x" << std::hex << (unsigned int)val;
              if (c + 1 < size) out << ", ";
              if (c % 32 == 31) out << std::endl << "\t";
            }
            out << std::endl << "}; " << std::endl << std::endl;
          } 
        else if (!run_cmd->kernel->arg_is_local[i])
          {
            /* Scalars are stored to global buffers automatically. Dump them to buffers. */
            unsigned start_addr = BSWAP(dev_cmd.args[i]);
            unsigned size = al->size;

            out << "__global__ char scalar_arg_" << std::dec << i 
                << "[] = {" << std::endl << "\t";

            MemorySystem &mems = simulator.memorySystem();
            MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
            for (std::size_t c = 0; c < size; ++c) {
              unsigned char val = globalMem->read (start_addr + c);
              out << "0x" << std::hex << (unsigned int)val;
              if (c + 1 < size) out << ", ";
              if (c % 32 == 31) out << std::endl << "\t";
            }
            out << std::endl << "}; " << std::endl << std::endl;
          }
      }
    
    /* Setup the kernel command initialization values, pointing to the
       global buffers for the buffer arguments, and using the original values
       for the rest. */       

    TCEString kernelMdSymbolName = "_";
    kernelMdSymbolName += run_cmd->kernel->name;
    kernelMdSymbolName += "_md";

    out << "extern __global__ __kernel_metadata " << kernelMdSymbolName << ";" << std::endl << std::endl;

    out << "kernel_exec_cmd kernel_command = {" << std::endl
        << "\t.status = " << std::hex << "(uint32_t)0x" << BSWAP(dev_cmd.status) 
        << "," << std::endl
        << "\t.work_dim = " << std::dec << BSWAP(dev_cmd.work_dim) 
        << ", " << std::endl
        << "\t.num_groups = {" 
        << std::dec << BSWAP(dev_cmd.num_groups[0]) << ", "
        << std::dec << BSWAP(dev_cmd.num_groups[1]) << ", "
        << std::dec << BSWAP(dev_cmd.num_groups[2]) << "}," << std::endl
        << "\t.global_offset = {"
        << std::dec << BSWAP(dev_cmd.global_offset[0]) << ", "
        << std::dec << BSWAP(dev_cmd.global_offset[1]) << ", "
        << std::dec << BSWAP(dev_cmd.global_offset[2]) << "}" << std::endl
        << "};" << std::endl;

    out << std::endl;
    out << "__attribute__((noinline))" << std::endl;
    out << "void initialize_kernel_launch() {" << std::endl;

    out << "\tkernel_command.kernel = (uint32_t)&" << kernelMdSymbolName << ";" << std::endl;
    int a = 0;
    for (; a < run_cmd->kernel->num_args + run_cmd->kernel->num_locals; ++a)
      {
        struct pocl_argument *al = &(run_cmd->arguments[a]);
        out << "\tkernel_command.args[" << std::dec << a << "] = ";
        
        if (run_cmd->kernel->arg_is_local[a] || a >= run_cmd->kernel->num_args)
          {
            /* Local buffers are managed by the host so the local
               addresses are already valid. */
            out << "(uint32_t)" << "0x" << std::hex << BSWAP(dev_cmd.args[a]);
          }
        else if (run_cmd->kernel->arg_is_pointer[a] && dev_cmd.args[a] != 0)
          {
            unsigned start_addr = 
              ((chunk_info_t*)((*(cl_mem *) (al->value))->device_ptrs[parent->dev_id].mem_ptr))->start_address;
            
            out << "(uint32_t)&buffer_" << std::hex << start_addr << "[0]";
          }
        else 
          {
            /* Scalars have been stored to global memory automatically. Point
               to the generated buffers.
             */
            out << "(uint32_t)&scalar_arg_" << std::dec << a;
          }
        out << ";" << std::endl;
    }   
    
    //    out << "\tlwpr_print_str(\"tta: initialized the standalone kernel lauch\\n\");" << std::endl;
    out << "}" << std::endl;     
    out.close();

    // Create the build script.

    std::ofstream scriptout(buildScriptFname.c_str());
    scriptout 
        << tceccCommandLine(run_cmd, fname + " " + tempDir + "/parallel.bc", 
                            "standalone.tpef", " -D_STANDALONE_MODE=1");
    scriptout.close();

    ++runCounter;
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
  /* If set to true, extra files are produced that can be used to
     reproduce execution of a single kernel outside pocl. The produced
     files contain all the data needed to execute a single kernel
     launch that was done in the host program using the 
     clEnqueueNDRangeKernel(). The files are produced to the temp
     directory of each final kernel binary. */
  bool produceStandAloneProgram_;
  /* This stream is pointing to the memory initialization function for the
     produced standalone kernel. All host-device writes are reproduced
     as C code in this function. */
  std::ofstream* standaloneProgramInitFunc_;
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


cl_ulong
pocl_ttasim_get_timer_value (void *data) 
{
  TTASimDevice *d = (TTASimDevice*)data;
  return d->timeStamp();
}
