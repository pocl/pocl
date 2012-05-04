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

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
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

using namespace TTAMachine;

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128

#define ALIGNMENT (max(ALIGNOF_FLOAT16, ALIGNOF_DOUBLE16))

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
  /* The ttasim engine handle. */
  SimpleSimulatorFrontend *simulator;
  /* The bufalloc memory regions for memory allocation book keeping. */
  struct memory_region local_mem;
  struct memory_region global_mem;
  
  AddressSpace *local_as;
  AddressSpace *global_as;
  char* machine_file;
};

size_t pocl_ttasim_max_work_item_sizes[] = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX};

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
  d->simulator = simFront;
  /* TODO:
     d->simulator->setZeroFillMemoriesOnReset(false);
  */
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
  device->global_mem_size = d->global_as->end() - d->local_as->start();

  init_mem_region 
    (&d->local_mem, (memory_address_t)d->local_as->start(), device->local_mem_size);
  init_mem_region 
    (&d->global_mem, (memory_address_t)d->global_as->start(), device->global_mem_size);
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
         back to the host's memory in case it's used as an output. */
      pocl_ttasim_copy_h2d (d, host_ptr, chunk, size);
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
     
      /* TODO: add the launcher code + main */
      /* At this point the kernel has been fully linked. */
      std::string buildCmd = 
        "tcecc " + deviceMainSrc + " " + bytecode + " -a " + d->machine_file + 
        " -O3 -o " + assemblyFileName;
      error = system(buildCmd.c_str());
      if (error != 0)
        POCL_ABORT("Error while running tcecc.");
    }

  /* Load the binary assembly (TPEF) to the simulator. */
  d->simulator->loadProgram(assemblyFileName);
  d->simulator->run();
}

/* Copy data between the host memory and the global memory of the device. */
void 
pocl_ttasim_copy_h2d (void *device_data, const void *src_ptr, chunk_info_t *dest, size_t count)  
{
  /* Find the simulation model for the global address space. */
  struct data *d = (struct data*)device_data;
  MemorySystem &mems = d->simulator->memorySystem();
  Memory& globalMem = mems.memory (*d->global_as);
  for (int i = 0; i < count; ++i)
    globalMem.write (dest->start_address + i, ((Memory::MAU*)src_ptr)[i]);
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
