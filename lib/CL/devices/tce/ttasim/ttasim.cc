/* ttasim.cc - a pocl device driver for simulating TTA devices using TCE's
   ttasim

   Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University

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
#include "common.h"
#include "common_driver.h"
#include "pocl_device.h"
#include "pocl_util.h"
#include "common.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <string>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

/* Supress some warnings because of including tce_config.h after pocl's config.h. */
#undef LLVM_VERSION

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
#include "devices.h"
#include "pocl_file_util.h"
#include "pocl_hash.h"

#include "common_driver.h"

#define DEFAULT_WG_SIZE 64

using namespace TTAMachine;

static void *pocl_ttasim_thread (void *p);

void
pocl_ttasim_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "ttasim";

  ops->probe = pocl_ttasim_probe;
  ops->uninit = pocl_ttasim_uninit;
  ops->reinit = pocl_ttasim_reinit;
  ops->init = pocl_ttasim_init;
  ops->init_build = pocl_tce_init_build;

  ops->alloc_mem_obj = pocl_tce_alloc_mem_obj;
  ops->free = pocl_tce_free;

  ops->read = pocl_tce_read;
  ops->read_rect = pocl_tce_read_rect;
  ops->write = pocl_tce_write;
  ops->write_rect = pocl_tce_write_rect;
  ops->copy = pocl_tce_copy;
  ops->copy_rect = pocl_tce_copy_rect;
  ops->map_mem = pocl_tce_map_mem;
  ops->unmap_mem = pocl_tce_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;
  ops->run = pocl_tce_run;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->compile_kernel = pocl_tce_compile_kernel;

  // new driver api
  ops->join = pocl_tce_join;
  ops->submit = pocl_tce_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_tce_notify;
  ops->flush = pocl_tce_flush;
  ops->wait_event = pocl_tce_wait_event;
  ops->free_event_data = pocl_tce_free_event_data;
  ops->notify_cmdq_finished = pocl_tce_notify_cmdq_finished;
  ops->notify_event_finished = pocl_tce_notify_event_finished;
  ops->build_hash = pocl_tce_build_hash;
  ops->get_device_info_ext = NULL;

  ops->init_queue = pocl_tce_init_queue;
  ops->free_queue = pocl_tce_free_queue;
}


unsigned int
pocl_ttasim_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  if(env_count < 0)
    return 0;

  return env_count;
}

/* This is a version number that should be increased when there is
 * a change in TCE drivers that makes previous pocl-binaries incompatible
 * (e.g. a change in generated device image file names, etc) */
#define POCL_TCE_BINARY_VERSION "1"

int
pocl_tce_device_hash (const char *adf_file, const char *llvm_triplet,
                             char *output)
{

  SHA1_CTX ctx;
  uint8_t bin_dig[SHA1_DIGEST_SIZE];

  char *content;
  uint64_t size;
  int err = pocl_read_file (adf_file, &content, &size);
  if (err || (content == NULL) || (size == 0))
    POCL_ABORT ("Can't find ADF file %s \n", adf_file);

  pocl_SHA1_Init (&ctx);
  pocl_SHA1_Update (&ctx, (const uint8_t *)POCL_TCE_BINARY_VERSION, 1);
  pocl_SHA1_Update (&ctx, (const uint8_t *)llvm_triplet, strlen (llvm_triplet));
  pocl_SHA1_Update (&ctx, (const uint8_t *)content, size);

  if (pocl_is_option_set ("POCL_TCECC_EXTRA_FLAGS"))
    {
      const char *extra_flags
          = pocl_get_string_option ("POCL_TCECC_EXTRA_FLAGS", "");
      pocl_SHA1_Update (&ctx, (const uint8_t *)extra_flags, strlen (extra_flags));
    }

  pocl_SHA1_Final (&ctx, bin_dig);

  unsigned i;
  for (i = 0; i < SHA1_DIGEST_SIZE; i++)
    {
      *output++ = (bin_dig[i] & 0x0F) + 65;
      *output++ = ((bin_dig[i] & 0xF0) >> 4) + 65;
    }
  *output = 0;
  return 0;
}


class TTASimDevice : public TCEDevice {
public:
  TTASimDevice(cl_device_id dev, const char *adfName)
      : TCEDevice(dev, adfName), simulator(adfName, false),
        simulatorCLI(simulator.frontend()), debuggerRequested(false) {

    char dev_name[256];

    const char *adf = strrchr(adfName, '/');
    if (adf != NULL) adf++;
    if (snprintf (dev_name, 256, "ttasim-%s", adf) < 0)
      POCL_ABORT("Unable to generate the device name string.\n");
    dev->long_name = strdup(dev_name);  
    ++device_count;

    SigINTHandler* ctrlcHandler = new SigINTHandler(this);
    Application::setSignalHandler(SIGINT, *ctrlcHandler);

    produceStandAloneProgram_ =
        pocl_get_bool_option("POCL_TCE_STANDALONE", 0) != 0;

    setMachine(simulator.machine());
    if (machine_->isLittleEndian()) {
      dev->endian_little = CL_TRUE;
      dev->llvm_target_triplet = "tcele-tut-llvm";
    } else {
      dev->endian_little = CL_FALSE;
      dev->llvm_target_triplet = "tce-tut-llvm";
    }

#if defined(WORDS_BIGENDIAN) && WORDS_BIGENDIAN == 1
    needsByteSwap = ((dev->endian_little == CL_TRUE) ? true : false);
#else
    needsByteSwap = ((dev->endian_little == CL_TRUE) ? false : true);
#endif

    SHA1_digest_t digest;
    pocl_tce_device_hash(adfName, dev->llvm_target_triplet,
                         (char *)digest);
    this->build_hash = (char *)digest;

    initMemoryManagement(simulator.machine());

    POCL_INIT_LOCK(lock);
    POCL_INIT_COND(simulation_start_cond);
    PTHREAD_CHECK(
        pthread_create(&ttasim_thread, NULL, pocl_ttasim_thread, this));
    PTHREAD_CHECK(
        pthread_create(&driver_thread, NULL, pocl_tce_driver_thread, this));
  }

  ~TTASimDevice() {
    shutdownRequested = true;
    simulator.stop();
    POCL_LOCK(lock);
    POCL_SIGNAL_COND(simulation_start_cond);
    POCL_UNLOCK(lock);
    POCL_JOIN_THREAD(ttasim_thread);

    POCL_FAST_LOCK(wq_lock);
    POCL_SIGNAL_COND(wakeup_cond);
    POCL_FAST_UNLOCK(wq_lock);
    POCL_JOIN_THREAD(driver_thread);
  }

  virtual void copyHostToDevice(const void *host_ptr, uint32_t dest_addr, size_t count) {
    MemorySystem &mems = simulator.memorySystem();
    MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
    for (std::size_t i = 0; i < count; ++i) {
      unsigned char val = ((char*)host_ptr)[i];
      globalMem->write (dest_addr + i, (Memory::MAU)(val));
    }
  }

  virtual void copyDeviceToDevice(uint32_t src_addr, uint32_t dest_addr, size_t count) {
    MemorySystem &mems = simulator.memorySystem();
    MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
    for (std::size_t i = 0; i < count; ++i) {
      Memory::MAU val = globalMem->read(src_addr + i);
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
    POCL_LOCK(lock);
    POCL_SIGNAL_COND(simulation_start_cond);
    POCL_UNLOCK(lock);
  }

  virtual void notifyKernelRunCommandSent(__kernel_exec_cmd &dev_cmd,
                                          _cl_command_run *run_cmd,
                                          uint32_t *gmem_ptr_positions,
                                          uint32_t gmem_count) {
    if (!produceStandAloneProgram_) return;

    static int runCounter = 0;
    TCEString tempDir((const char*)run_cmd->device_data);
    TCEString baseFname = tempDir + "/";
    baseFname << "standalone_" << runCounter;

    TCEString buildScriptFname = baseFname + "_build";
    TCEString fname = baseFname + ".c";
    TCEString parallel_bc = tempDir + "/parallel.bc";

    std::ofstream out(fname.c_str());

    out << "#include <pocl_device.h>" << std::endl << std::endl;

    out << "#undef ALIGN4" << std::endl;
    out << "#define ALIGN4 __attribute__ ((aligned (4)))" << std::endl;
    out << "#define __local__ __attribute__((address_space(" <<  TTA_ASID_LOCAL<< ")))" << std::endl;
    out << "#define __global__ __attribute__((address_space(" << TTA_ASID_GLOBAL << ")))" << std::endl;
    out << "#define __constant__ __attribute__((address_space("
        << TTA_ASID_CONSTANT << ")))" << std::endl
        << std::endl;
    out << "typedef volatile __global__ __kernel_exec_cmd kernel_exec_cmd;" << std::endl;

    /* Need to byteswap back as we are writing C code. */
#define BSWAP(__X) pocl_byteswap_uint32_t(__X, needsByteSwap)

    /* The standalone binary shall have the same input data as in the original
       kernel host-device kernel launch command. The data is put into
       initialized global arrays to easily exclude the initialization time from
       the execution time. Otherwise, the same command data is used for
       reproducing the execution. For example, the local memory allocations
       (addresses) are the same as in the original one. */

    /* Create the global buffers along with their initialization data. */
    cl_kernel kernel = run_cmd->kernel;
    pocl_kernel_metadata_t *meta = kernel->meta;
    uint32_t total_gmems = 0;
    /* store addresses used for buffer names later*/
    uint32_t gmem_startaddrs[1024];

    for (size_t i = 0; i < meta->num_args; ++i)
      {
        struct pocl_argument *al = &(run_cmd->arguments[i]);
        if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
          {
            if (al->value == NULL) continue;

            cl_mem m = *(cl_mem *)(al->value);
            void *p = m->device_ptrs[parent->global_mem_id].mem_ptr;
            chunk_info_t *ci = (chunk_info_t *)p;
            unsigned start_addr = ci->start_address + al->offset;
            unsigned size = ci->size;

            out << "__global__ ALIGN4 char buffer_" << std::hex << start_addr
                << "[] = {\n";

            gmem_startaddrs[total_gmems++] = start_addr;

            MemorySystem &mems = simulator.memorySystem();
            MemorySystem::MemoryPtr globalMem = mems.memory (*global_as);
            for (std::size_t c = 0; c < size; ++c) {
              if (c % 32 == 0)
                out << std::endl << "\t";
              unsigned char val = globalMem->read (start_addr + c);
              out << "0x" << std::hex << (unsigned int)val;
              if (c + 1 < size) out << ", ";
            }
            out << std::endl << "};" << std::endl << std::endl;
          }
      }
      assert(total_gmems == gmem_count);

      /* Scalars (+addresses of global/local buffers) are stored to a single
       * global buffer. */
      {
        unsigned start_addr = BSWAP(dev_cmd.args);
        unsigned size = BSWAP(dev_cmd.args_size);

        out << "__global__ ALIGN4 char arg_buffer[] = {" << std::endl << "\t";

        MemorySystem &mems = simulator.memorySystem();
        MemorySystem::MemoryPtr globalMem = mems.memory(*global_as);
        for (std::size_t c = 0; c < size; ++c) {
          unsigned char val = globalMem->read(start_addr + c);
          out << "0x" << std::hex << (unsigned int)val;
          if (c + 1 < size)
            out << ", ";
          if (c % 32 == 31)
            out << std::endl << "\t";
        }
        out << std::endl << "};" << std::endl << std::endl;
      }

      /* pocl_context is stored in a global buffer too; copy it. */
      {
        unsigned start_addr = BSWAP(dev_cmd.ctx);
        unsigned size = BSWAP(dev_cmd.ctx_size);

        out << "__global__ ALIGN4 char ctx_buffer[] = {" << std::endl << "\t";

        MemorySystem &mems = simulator.memorySystem();
        MemorySystem::MemoryPtr globalMem = mems.memory(*global_as);
        for (std::size_t c = 0; c < size; ++c) {
          unsigned char val = globalMem->read(start_addr + c);
          out << "0x" << std::hex << (unsigned int)val;
          if (c + 1 < size)
            out << ", ";
          if (c % 32 == 31)
            out << std::endl << "\t";
        }
        out << std::endl << "};" << std::endl << std::endl;
      }

      /* Setup the kernel command initialization values, pointing to the
         global buffers for the buffer arguments, and using the original values
         for the rest. */

      TCEString kernelMdSymbolName = "_";
      kernelMdSymbolName += run_cmd->kernel->name;
      kernelMdSymbolName += "_md";

      out << "extern __global__ __kernel_metadata " << kernelMdSymbolName << ";"
          << std::endl
          << std::endl;

      out << "ALIGN4 kernel_exec_cmd kernel_command = {" << std::endl
          << "\t.status = " << std::hex << "(uint32_t)0x"
          << BSWAP(dev_cmd.status) << ",\n"
          << "\t.args_size = " << std::hex << "(uint32_t)0x"
          << BSWAP(dev_cmd.args_size) << ",\n"
          << "\t.ctx_size = " << std::hex << "(uint32_t)0x"
          << BSWAP(dev_cmd.ctx_size) << ",\n"
          << "};" << std::endl;

      out << std::endl;
      out << "__attribute__((noinline))" << std::endl;
      out << "void initialize_kernel_launch() {" << std::endl;

      // update the args and ctx pointers with actual addresses
      out << "\tkernel_command.args = (uint32_t)&arg_buffer;\n";
      out << "\tkernel_command.ctx = (uint32_t)&ctx_buffer;\n";
      out << "\tkernel_command.kernel_meta = (uint32_t)&" << kernelMdSymbolName
          << ";\n";

      out << "\tchar* kernargs = (char*)kernel_command.args;\n";
      out << "\tchar* temp = 0;\n\tuint32_t *tmp = 0;\n";
      for (size_t i = 0; i < gmem_count; ++i) {
        out << "\ttemp = kernargs + " << std::dec << gmem_ptr_positions[i]
            << ";\n";
        out << "\ttmp = (uint32_t*)temp;\n";
        out << "\t*tmp = (uint32_t)&buffer_" << std::hex << gmem_startaddrs[i]
            << "[0]";
        out << ";" << std::endl;
      }

      out << "}" << std::endl;
      out.close();

      // Create the build script.
      TCEString inputFiles = fname + " " + parallel_bc;
      std::ofstream scriptout(buildScriptFname.c_str());
      scriptout << tceccCommandLine(run_cmd, tempDir, inputFiles.c_str(),
                                    "standalone.tpef", " -D_STANDALONE_MODE=1");
      scriptout.close();

      ++runCounter;
  }


  SimpleSimulatorFrontend simulator;
  /* A Command Line Interface for debugging. */
  SimulatorCLI simulatorCLI;
  volatile bool debuggerRequested;

  pocl_thread_t ttasim_thread;
  pocl_thread_t driver_thread;
  pocl_cond_t simulation_start_cond;
  pocl_lock_t lock;

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
    if (d->shutdownRequested)
      goto EXIT;
    POCL_LOCK(d->lock);
    PTHREAD_CHECK(pthread_cond_wait(&d->simulation_start_cond, &d->lock));
    POCL_UNLOCK(d->lock);
    if (d->shutdownRequested)
      goto EXIT;

    d->simulator.run();

    if (d->debuggerRequested) {
      d->debuggerRequested = false;
      d->simulatorCLI.run();
      continue;
    }

    if (d->simulator.hadRuntimeError()) {
      //            POCL_MSG_ERR ("Runtime error in a ttasim device! Launching
      //            debugger CLI \n"); d->simulatorCLI.run();
      POCL_ABORT("Runtime error in a ttasim device.\n");
    }

  } while (true);

EXIT:
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

static const char *tta_native_device_aux_funcs[] = {"_pocl_memcpy_1",
                                                    "_pocl_memcpy_4", NULL};

static const char *tce_params[256];

cl_int
pocl_ttasim_init (unsigned j, cl_device_id dev, const char* parameters)
{
  if (parameters == NULL)
    POCL_ABORT("The tta device requires the adf file as a device parameter.\n"
               "Set it with POCL_TTASIMn_PARAMETERS=\"path/to/themachine.adf\".\n");

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->max_compute_units = 1;
  dev->max_work_item_dimensions = 3;
  dev->device_side_printf = 0;
  tce_params[j] = parameters;

  int max_wg
      = pocl_get_int_option ("POCL_MAX_WORK_GROUP_SIZE", DEFAULT_WG_SIZE);
  assert (max_wg > 0);
  max_wg = std::min (max_wg, DEFAULT_WG_SIZE);
  if (max_wg < 0)
    max_wg = DEFAULT_WG_SIZE;

  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1]
      = dev->max_work_item_sizes[2] = dev->max_work_group_size = max_wg;

  dev->preferred_wg_size_multiple = 8;
  dev->preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->preferred_vector_width_short
      = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->preferred_vector_width_float
      = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->preferred_vector_width_double
      = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  /* TODO: figure out what the difference between preferred and native widths
   * are. */
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

  dev->available = CL_TRUE;
#ifdef ENABLE_LLVM
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
#endif
  dev->spmd = CL_FALSE;
  dev->workgroup_pass = CL_TRUE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->queue_properties = CL_QUEUE_PROFILING_ENABLE;
  dev->vendor = "TTA-Based Co-design Environment";
  dev->profile = "EMBEDDED_PROFILE";
  dev->extensions = TCE_DEVICE_EXTENSIONS;

  dev->has_64bit_long = 1;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;

  dev->global_as_id = TTA_ASID_GLOBAL;
  dev->local_as_id = TTA_ASID_LOCAL;
  dev->constant_as_id = TTA_ASID_CONSTANT;
  dev->args_as_id = TTA_ASID_GLOBAL;
  dev->context_as_id = TTA_ASID_GLOBAL;

  SETUP_DEVICE_CL_VERSION (TCE_DEVICE_CL_VERSION_MAJOR,
                           TCE_DEVICE_CL_VERSION_MINOR);

  dev->parent_device = NULL;
  // ttasim does not support partitioning
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->partition_properties = (cl_device_partition_property *)calloc (
      dev->num_partition_properties, sizeof (cl_device_partition_property));
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  dev->mem_base_addr_align =
      dev->min_data_type_align_size = 16;

  dev->data = new TTASimDevice(dev, parameters);

  dev->arg_buffer_launcher = CL_TRUE;
  dev->grid_launcher = CL_FALSE;
  dev->device_aux_functions = tta_native_device_aux_funcs;

  dev->mem_base_addr_align = 128;
  dev->min_data_type_align_size = 128;

  return CL_SUCCESS;
}

cl_int
pocl_ttasim_uninit (unsigned j, cl_device_id device)
{
  POCL_MSG_PRINT_TCE("DEV UNINIT \n");
  delete (TTASimDevice *)device->data;
  return CL_SUCCESS;
}

cl_int pocl_ttasim_reinit(unsigned j, cl_device_id device) {
  POCL_MSG_PRINT_TCE("DEV REINIT \n");
  device->data = new TTASimDevice(device, tce_params[j]);
  return CL_SUCCESS;
}
