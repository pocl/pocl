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

#include <SimpleSimulatorFrontend.hh>
#include <Machine.hh>
#include <MemorySystem.hh>
#include <Program.hh>
#include <GlobalScope.hh>
#include <DataLabel.hh>
#include <SimulatorCLI.hh>
#include <SimulationEventHandler.hh>
#include <Listener.hh>

using namespace TTAMachine;

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128

#define ALIGNMENT (max(ALIGNOF_FLOAT16, ALIGNOF_DOUBLE16))

#define DEBUG_TTASIM_DRIVER

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
  /* The ttasim engine handle. */
  SimpleSimulatorFrontend *simulator;
  /* A Command Line Interface for debugging. */
  SimulatorCLI* simulatorCLI;
  volatile bool debuggerRequested;

  /* The bufalloc memory regions for memory allocation book keeping. */
  struct memory_region local_mem;
  struct memory_region global_mem;
  
  AddressSpace *local_as;
  AddressSpace *global_as;
  char* machine_file;

  pthread_t ttasim_thread;
  pthread_cond_t simulation_start_cond;
  pthread_mutex_t lock;

  cl_device_id parent;
};

size_t pocl_ttasim_max_work_item_sizes[] = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX};

uint32_t pocl_ttasim_read_device (void *device_data, uint32_t addr);
void pocl_ttasim_write_device (void *device_data, uint32_t addr, uint32_t word);

static 
void * 
pocl_ttasim_thread (void *p)
{
  struct data *d = (data*)p;
  do {
    pthread_cond_wait (&d->simulation_start_cond, &d->lock);
    std::cout << "### run!" << std::flush << std::endl;
    do {
      d->simulator->run();
      if (d->debuggerRequested) {
          d->debuggerRequested = false;
          d->simulatorCLI->run();
          continue;
      }

      if (d->simulator->hadRuntimeError()) {
          d->simulatorCLI->run();
          POCL_ABORT("Runtime error in a ttasim device.");
      }
    } while (true);
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
    SigINTHandler(struct data* d) :
        d_(d) {
    }

    /**
     * Stops the simulation.
     */
    virtual void execute(int /*data*/, siginfo_t* /*info*/) {
        std::cerr << "### ctrl-C handler" << std::endl;
        d_->debuggerRequested = true;
        d_->simulator->stop();
        std::cerr << "### handler exit" << std::endl;
        /* Make it a one shot handler. Next Ctrl-C should kill the program. */
        Application::restoreSignalHandler(SIGINT);
    }
private:
    struct data* d_;
};

void
pocl_ttasim_init (cl_device_id device, const char* parameters)
{
  static int device_count = 0;
  char dev_name[256];
  struct data *d;
  
  if (parameters == NULL)
    POCL_ABORT("The ttasim device requires the adf file as a device parameter.\n"
               "Set it with POCL_DEVICEn_PARAMETERS=\"path/to/themachine.adf\".\n");

  d = (struct data *) malloc (sizeof (struct data));
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;
  d->parent = device;

  device->data = d;

  if (device_count > 0)
    {
      if (snprintf (dev_name, 256, "ttasim%d", device_count) < 0)
        POCL_ABORT("Unable to generate the device name string.");
      device->name = strdup(dev_name);  
    }
  ++device_count;

  SimpleSimulatorFrontend *simFront = 
    new SimpleSimulatorFrontend(parameters);  

  /* TODO: only if debug_mode is on. */
#if 1
  SigINTHandler* ctrlcHandler = new SigINTHandler(d);
  Application::setSignalHandler(SIGINT, *ctrlcHandler);
#endif
  d->debuggerRequested = false;

  d->simulator = simFront;
  d->simulatorCLI = new SimulatorCLI(simFront->frontend());
  d->machine_file = strdup(parameters);

  /* Create the memory allocation book keeping structures based on
     the machine's address spaces (see tta.txt). */
  const Machine& mach = simFront->machine();
  AddressSpace *local = NULL, *global = NULL;
  Machine::AddressSpaceNavigator nav = mach.addressSpaceNavigator();
  for (int i = 0; i < nav.count(); ++i) 
    {
      AddressSpace *as = nav.item(i);
      if (as->hasNumericalId(TTA_ASID_PRIVATE) &&
          as->hasNumericalId(TTA_ASID_LOCAL))
        {
          d->local_as = as;
          continue;
        } 
      if (as->hasNumericalId(TTA_ASID_GLOBAL) &&
          as->hasNumericalId(TTA_ASID_CONSTANT))
        {
          d->global_as = as;
        }
    }
  if (d->local_as == NULL) 
    POCL_ABORT("local address space not found in the ADF. Mark it by adding numerical ids 0 and 4 to the AS.\n");
  if (d->global_as == NULL) 
    POCL_ABORT("global address space not found in the ADF. Mark it by adding numerical ids 3 and 5 to the AS.\n");

  int local_size = 
    d->local_as->end() - d->local_as->start() - TTA_UNALLOCATED_LOCAL_SPACE;
  if (local_size < 0)
    POCL_ABORT("Not enough space in the local memory with the assumed unallocated space.\n");

  device->local_mem_size = local_size;
  device->global_mem_size = d->global_as->end() - d->local_as->start() - TTA_UNALLOCATED_GLOBAL_SPACE;
  if (device->global_mem_size < 0)
    POCL_ABORT("Not enough space in the global memory with the assumed unallocated space.\n");

  init_mem_region 
    (&d->local_mem, (memory_address_t)d->local_as->start(), device->local_mem_size);
  init_mem_region 
    (&d->global_mem, (memory_address_t)d->global_as->start() + TTA_UNALLOCATED_GLOBAL_SPACE, 
     device->global_mem_size);

  pthread_cond_init (&d->simulation_start_cond, NULL);

  pthread_create (&d->ttasim_thread, NULL, pocl_ttasim_thread, d);
}

void *
pocl_ttasim_malloc (void *device_data, cl_mem_flags flags,
		    size_t size, void *host_ptr)
{
  void *b;
  struct data* d = (struct data*)device_data;

  chunk_info_t *chunk = alloc_buffer (&d->global_mem, size);
  if (chunk == NULL) return NULL;

  if ((flags & CL_MEM_COPY_HOST_PTR) ||  
      ((flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL))
    {
      /* TODO: 
         CL_MEM_USE_HOST_PTR must synch the buffer after execution 
         back to the host's memory in case it's used as an output (?). */
      pocl_ttasim_copy_h2d (d, host_ptr, chunk->start_address, size);
      return (void*) chunk;
    }
  return (void*) chunk;
}

void
pocl_ttasim_free (void *data, cl_mem_flags flags, void *ptr)
{
  if (flags & CL_MEM_USE_HOST_PTR)
    POCL_ABORT_UNIMPLEMENTED();

  free_chunk ((chunk_info_t*) ptr);
}

void
pocl_ttasim_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  POCL_ABORT_UNIMPLEMENTED();
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
}

void
pocl_ttasim_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  POCL_ABORT_UNIMPLEMENTED();
  if (host_ptr == device_ptr)
    return;

  memcpy (device_ptr, host_ptr, cb);
}

void
pocl_ttasim_run 
(void *data, const char *tmpdir,
 cl_kernel kernel,
 struct pocl_context *pc)
{
  struct data *d;
  int error;
  char bytecode[POCL_FILENAME_LENGTH];
  char assembly[POCL_FILENAME_LENGTH];
  char module[POCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  unsigned device;
  struct pocl_argument *p;
  size_t x, y, z;
  unsigned i;
  pocl_workgroup w;

  assert (data != NULL);
  d = (struct data *) data;

  std::string assemblyFileName(tmpdir);
  assemblyFileName += "/parallel.tpef";

  std::string kernelMdSymbolName = "_";
  kernelMdSymbolName += kernel->name;
  kernelMdSymbolName += "_md";

  if (access (assemblyFileName.c_str(), F_OK) != 0)
    {
      error = snprintf (bytecode, POCL_FILENAME_LENGTH,
                        "%s/linked.bc", tmpdir);
      assert (error >= 0);
      
      error = snprintf (command, COMMAND_LENGTH,
			LLVM_LD " -link-as-library -o %s %s/%s",
                        bytecode, tmpdir, POCL_PARALLEL_BC_FILENAME);
      assert (error >= 0);
      
      error = system(command);
      assert (error == 0);
      
      std::string deviceMainSrc = "";

      if (access (BUILDDIR "/lib/CL/devices/ttasim/tta_device_main.c", R_OK) == 0)
        deviceMainSrc = BUILDDIR "/lib/CL/devices/ttasim/tta_device_main.c";
      else
        POCL_ABORT_UNIMPLEMENTED();
     
      std::string kernelObjSrc = "";
      kernelObjSrc += tmpdir;
      kernelObjSrc += "/../descriptor.so.kernel_obj.c";

      /* TODO: add the launcher code + main */
      /* At this point the kernel has been fully linked. */
      std::string buildCmd = 
        std::string("tcecc -llwpr -I ") + SRCDIR + "/include " + deviceMainSrc + " " + 
        kernelObjSrc + " " + bytecode + " -a " + d->machine_file + 
        " -k " + kernelMdSymbolName +
        " --swfp -g -O0 -o " + assemblyFileName;
      std::cerr << "CMD: " << buildCmd << std::endl;
      error = system(buildCmd.c_str());
      if (error != 0)
        POCL_ABORT("Error while running tcecc.");
    }

  /* Load the binary assembly (TPEF) to the simulator. */
  if (d->simulator->isRunning()) 
    d->simulator->stop();
  while (d->simulator->isRunning()) 
    ;
  d->simulator->loadProgram(assemblyFileName);
  pthread_cond_signal (&d->simulation_start_cond);

  /* Figure out the locations of the shared data structures in
     the device memories from the fully-linked program. 
     
     TODO: load both the ADF and the TPEF objects only once and use the
     object interface of the simulator instead of the file name interface.  
  */
  TTAProgram::Program* prog = 
    TTAProgram::Program::loadFromTPEF(assemblyFileName, *d->local_as->machine());
  assert (prog != NULL);

  const TTAProgram::GlobalScope& globalScope = prog->globalScopeConst();
  
  uint32_t kernelAddr;
  uint32_t commandQueueAddr;
  try {
    kernelAddr = globalScope.dataLabel(kernelMdSymbolName).address().location();
    commandQueueAddr = globalScope.dataLabel("kernel_command").address().location();
  } catch (const KeyNotFound& e) {
    POCL_ABORT ("Could not find the shared data structures from the device binary.");
  }
  //printf ("_test_kernel_md @ %u  kernel_command @ %u\n", kernelAddr, commandQueueAddr);

  int swap = 1; /* TODO, compare the host&device endiannesses */

  __kernel_exec_cmd cmd;
  cmd.kernel = byteswap_uint32_t (kernelAddr, swap);

  struct pocl_argument *al;  

  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(kernel->arguments[i]);
      if (kernel->arg_is_local[i])
        {
          cmd.dynamic_local_arg_sizes[i] = byteswap_uint32_t(al->size, swap);
        }
      else if (kernel->arg_is_pointer[i])
        cmd.args[i] = byteswap_uint32_t 
          (((chunk_info_t*)((*(cl_mem *) (al->value))->device_ptrs[d->parent->dev_id]))->start_address, swap);
      else /* The scalar values should be byteswapped by the user. */
        {
          /* Copy the scalar argument data to the shared memory. */
          chunk_info_t* arg_space = 
            (chunk_info_t*)pocl_ttasim_malloc (d, CL_MEM_COPY_HOST_PTR, al->size, al->value);
          printf ("host: copied value from %x to global argument memory\n", al->value);
          cmd.args[i] = byteswap_uint32_t (arg_space->start_address, swap);
        }
    }
  cmd.work_dim = byteswap_uint32_t (pc->work_dim, swap);
  cmd.num_groups[0] = byteswap_uint32_t (pc->num_groups[0], swap);
  cmd.num_groups[1] = byteswap_uint32_t (pc->num_groups[1], swap);
  cmd.num_groups[2] = byteswap_uint32_t (pc->num_groups[2], swap);

  cmd.global_offset[0] = byteswap_uint32_t (pc->global_offset[0], swap);
  cmd.global_offset[1] = byteswap_uint32_t (pc->global_offset[1], swap);
  cmd.global_offset[2] = byteswap_uint32_t (pc->global_offset[2], swap);

  cmd.status = byteswap_uint32_t (POCL_KST_FREE, swap);

#ifdef DEBUG_TTASIM_DRIVER
  printf("host: waiting for the device command queue (@ %x) to get room.\n",
         commandQueueAddr);
  printf("host: commmand queue status: %d\n",
         pocl_ttasim_read_device (d, commandQueueAddr));
#endif
  /* Wait until the device command queue has room. */
  do {} 
  while (pocl_ttasim_read_device (d, commandQueueAddr) != POCL_KST_FREE);

#ifdef DEBUG_TTASIM_DRIVER
  printf( "host: writing the command.\n");
#endif
  pocl_ttasim_copy_h2d (d, &cmd, commandQueueAddr, sizeof(__kernel_exec_cmd) );

  /* Ensure the READY status is written the last so the device doesn't
     start executing before all the cmd data has been written. We 
     need a flush or similar mechanism to ensure all the data has 
     been really written, in case the data transfers are not guaranteed
     to be ordered. */
  pocl_ttasim_write_device (d, commandQueueAddr, byteswap_uint32_t (POCL_KST_READY, swap));

#ifdef DEBUG_TTASIM_DRIVER
  printf("host: commmand queue status: %x\n",
         pocl_ttasim_read_device (d, commandQueueAddr));

  printf("host: waiting for the command to get executed.\n");
#endif
  /* Wait until the command has executed. */
  do {} 
  while (pocl_ttasim_read_device (d, commandQueueAddr) != POCL_KST_FINISHED);

#ifdef DEBUG_TTASIM_DRIVER
  printf( "host: done. Freeing the command queue entry.\n");
#endif
  /* We are done with this kernel, free the command queue entry. */
  pocl_ttasim_write_device (d, commandQueueAddr, byteswap_uint32_t (POCL_KST_FREE, swap));
}

/* Reads a single 32bit word from the device global memory. 

   NOTE: ttasim word read/write interface byteswaps internally. */
uint32_t
pocl_ttasim_read_device (void *device_data, uint32_t addr)  
{
  struct data *d = (struct data*)device_data;
  MemorySystem &mems = d->simulator->memorySystem();
  Memory& globalMem = mems.memory (*d->global_as);
  uint32_t val;
  globalMem.read (addr, 4, val);
  return val;
}

/* Writes a single 32bit word to the device global memory. 

   NOTE: ttasim word read/write interface byteswaps internally. */
void
pocl_ttasim_write_device (void *device_data, uint32_t addr, uint32_t word)  
{
  pocl_ttasim_copy_h2d (device_data, &word, addr, sizeof (word) );
}

/* Copy data between the host memory and the global memory of the device. 
    
   Raw byte copy without byte swapping. 
 */
void 
pocl_ttasim_copy_h2d (void *device_data, const void *src_ptr, uint32_t dest_addr, size_t count)  
{
  /* Find the simulation model for the global address space. */
  struct data *d = (struct data*)device_data;
  MemorySystem &mems = d->simulator->memorySystem();
  Memory& globalMem = mems.memory (*d->global_as);
  for (int i = 0; i < count; ++i) {
    unsigned char val = ((char*)src_ptr)[i];
    globalMem.write (dest_addr + i, (Memory::MAU)(val));
  }
}

void
pocl_ttasim_copy (void *data, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb)
{
  POCL_ABORT_UNIMPLEMENTED();
  if (src_ptr == dst_ptr)
    return;
  
  memcpy (dst_ptr, src_ptr, cb);
}

void
pocl_ttasim_copy_rect (void *data,
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
pocl_ttasim_write_rect (void *data,
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
pocl_ttasim_read_rect (void *data,
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


void *
pocl_ttasim_map_mem (void *data, void *buf_ptr, 
                    size_t offset, size_t size) 
{
  POCL_ABORT_UNIMPLEMENTED();

  /* All global pointers of the pthread/CPU device are in 
     the host address space already, and up to date. */     
  return buf_ptr + offset;
}

void
pocl_ttasim_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  free (d);
  device->data = NULL;
}
