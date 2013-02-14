/* cellspu.c - a pocl device driver for Cell SPU.

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

#include "cellspu.h"
#include "install-paths.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <../dev_image.h>
#include <sys/time.h>

#include <libspe2.h>
#include "pocl_device.h"

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 128

//#define DEBUG_CELLSPU_DRIVER

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
};

//TODO: global, or per-device?
spe_context_ptr_t spe_context;
//TODO: this certainly should be per-program (per kernel?)
spe_program_handle_t *hello_spu;
//TODO: again - not global...
memory_region_t spe_local_mem;

void
pocl_cellspu_init (cl_device_id device, const char* parameters)
{
  struct data *d;

  d = (struct data *) malloc (sizeof (struct data));
  device->data = d;
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->global_mem_size = 256*1024;
  device->max_mem_alloc_size = device->global_mem_size / 2;

  // TODO: find the API docs. what are the params?
  spe_context = spe_context_create(0,NULL);
  if (spe_context == NULL) perror("spe_context_create fails");
  
  // initialize the SPE local storage allocator. 
  init_mem_region( &spe_local_mem, CELLSPU_OCL_BUFFERS_START, device->max_mem_alloc_size); 

}

/* 
 * Allocate a chunk for kernel local variables.
 */
void *
cellspu_malloc_local (void *device_data, size_t size)
{
  struct data* d = (struct data*)device_data;
  chunk_info_t *chunk = alloc_buffer (&spe_local_mem, size);
  return (void*) chunk;

}
void *
pocl_cellspu_malloc (void *device_data, cl_mem_flags flags,
		     size_t size, void *host_ptr)
{
  void *b;
  struct data* d = (struct data*)device_data;

  //TODO: unglobalify spe_local_mem
  chunk_info_t *chunk = alloc_buffer (&spe_local_mem, size);
  if (chunk == NULL) return NULL;

#ifdef DEBUG_CELLSPU_DRIVER
  printf("host: malloc %x (host) %x (device) size: %u\n", host_ptr, chunk->start_address, size);
#endif
#if 0
  if ((flags & CL_MEM_COPY_HOST_PTR) ||  
      ((flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL))
    {
      /* TODO: 
         CL_MEM_USE_HOST_PTR must synch the buffer after execution 
         back to the host's memory in case it's used as an output (?). */
      d->copyHostToDevice(host_ptr, chunk->start_address, size);
      return (void*) chunk;
    }
#endif
  return (void*) chunk;

}

void
pocl_cellspu_free (void *data, cl_mem_flags flags, void *ptr)
{
  POCL_ABORT_UNIMPLEMENTED();

  if (flags & CL_MEM_USE_HOST_PTR)
    return;
  
  free (ptr);
}

void
pocl_cellspu_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
	chunk_info_t *chunk = (chunk_info_t*)device_ptr;
	assert( chunk->is_allocated  && "cellspu: writing to an ullacoated memory?");

#ifdef DEBUG_CELLSPU_DRIVER
	printf("cellspu: read %d bytes to %x (host) from %x (device)\n", cb, host_ptr,chunk->start_address);
#endif
	void *mmap_base=spe_ls_area_get( spe_context );
	memcpy( host_ptr, mmap_base+(chunk->start_address), cb);

}

/* write 'bytes' of bytes from *host_a to SPU local storage area. */
void cellspu_memwrite( void *lsa, const void *host_a, size_t bytes )
{	
#ifdef DEBUG_CELLSPU_DRIVER
	printf("cellspu: write %d bytes from %x (host) to %x (device)\n", bytes, host_a,lsa);
#endif
	void *mmap_base=spe_ls_area_get( spe_context );
	memcpy( (void*)(mmap_base+(int)lsa), (const void*)host_a, bytes);
}

void
pocl_cellspu_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
	chunk_info_t *chunk = (chunk_info_t*)device_ptr;
	assert( chunk->is_allocated  && "cellspu: writing to an ullacoated memory?");
        cellspu_memwrite( (void*)(chunk->start_address), host_ptr, cb );
}


void
pocl_cellspu_run 
(void *data, 
 _cl_command_node* cmd)
{
  struct data *d;
  int error;
  char bytecode[POCL_FILENAME_LENGTH];
  char assembly[POCL_FILENAME_LENGTH];
  char module[POCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  unsigned device;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  pocl_workgroup w;
  char* tmpdir = cmd->command.run.tmp_dir;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  const char* kern_func = kernel->function_name;
  unsigned int entry = SPE_DEFAULT_ENTRY;

  assert (data != NULL);
  d = (struct data *) data;

  error = snprintf 
    (module, POCL_FILENAME_LENGTH,
     "%s/parallel.so", tmpdir);
  assert (error >= 0);

  // This is the entry to the kenrel. We currently hard-code it
  // into the SPU binary. Resulting in only one entry-point per 
  // SPU image.
  // TODO: figure out which function to call given what conditions
  snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
            "_%s_workgroup_fast", kernel->function_name);


  if ( access (module, F_OK) != 0)
    {
      char *llvm_ld;
      error = snprintf (bytecode, POCL_FILENAME_LENGTH,
                        "%s/linked.bc", tmpdir);
      assert (error >= 0);
      
      if (getenv("POCL_BUILDING") != NULL)
        llvm_ld = BUILDDIR "/tools/llvm-ld/pocl-llvm-ld";
      else if (access(PKGLIBEXECDIR "/pocl-llvm-ld", X_OK) == 0)
        llvm_ld = PKGLIBEXECDIR "/pocl-llvm-ld";
      else
        llvm_ld = "pocl-llvm-ld";

      error = snprintf (command, COMMAND_LENGTH,
			"%s --disable-opt -link-as-library -o %s %s/%s",
                        llvm_ld, bytecode, tmpdir, POCL_PARALLEL_BC_FILENAME);
      assert (error >= 0);
      
      error = system(command);
      assert (error == 0);
      
      error = snprintf (assembly, POCL_FILENAME_LENGTH,
			"%s/parallel.s",
			tmpdir);
      assert (error >= 0);
      
      // "-relocation-model=dynamic-no-pic" is a magic string,
      // I do not know why it has to be there to produce valid
      // sos on x86_64
      error = snprintf (command, COMMAND_LENGTH,
			LLC " " HOST_LLC_FLAGS " -o %s %s",
			assembly,
			bytecode);
      assert (error >= 0);
      error = system (command);
      assert (error == 0);
           

      // Compile the assembly version of the OCL kernel with the
      // C wrapper to get a spulet
      error = snprintf (command, COMMAND_LENGTH,
			"spu-gcc lib/CL/devices/cellspu/spe_wrap.c -o %s %s "
			" -Xlinker --defsym -Xlinker _ocl_buffer=%d"
			" -Xlinker --defsym -Xlinker kernel_command=%d"
			" -I . -D_KERNEL=%s -std=c99",
			module,
			assembly, 
			CELLSPU_OCL_BUFFERS_START,
			CELLSPU_KERNEL_CMD_ADDR,
			workgroup_string);
      assert (error >= 0);
#ifdef DEBUG_CELLSPU_DRIVER
      printf("compiling: %s\n", command); fflush(stdout); 
#endif
      error = system (command);
      assert (error == 0);

    }
      
    // Load the SPU with the newly generated binary
    hello_spu = spe_image_open( (const char*)module );
    if( spe_program_load( spe_context, hello_spu) )
        perror("spe_program_load fails");
    
//
//  /* Find which device number within the context correspond
//     to current device.  */
//  for (i = 0; i < kernel->context->num_devices; ++i)
//    {
//      if (kernel->context->devices[i]->data == data)
//	{
//	  device = i;
//	  break;
//	}
//    }
//

  // This structure gets passed to the device.
  // It contains all the info needed to run a kernel  
  __kernel_exec_cmd dev_cmd;
  dev_cmd.work_dim = cmd->command.run.pc.work_dim;
  dev_cmd.num_groups[0] = cmd->command.run.pc.num_groups[0];
  dev_cmd.num_groups[1] = cmd->command.run.pc.num_groups[1];
  dev_cmd.num_groups[2] = cmd->command.run.pc.num_groups[2];

  dev_cmd.global_offset[0] = cmd->command.run.pc.global_offset[0];
  dev_cmd.global_offset[1] = cmd->command.run.pc.global_offset[1];
  dev_cmd.global_offset[2] = cmd->command.run.pc.global_offset[2];


  // the code below is lifted from pthreads :) 
  uint32_t *arguments = dev_cmd.args;

  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(kernel->dyn_arguments[i]);
      if (kernel->arg_is_local[i])
        {
          chunk_info_t* local_chunk = cellspu_malloc_local (d, al->size);
          if (local_chunk == NULL)
            POCL_ABORT ("Could not allocate memory for a local argument. Out of local mem?\n");

          dev_cmd.args[i] = local_chunk->start_address;

        }
      else if (kernel->arg_is_pointer[i])
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            arguments[i] = (uint32_t)NULL;
          else
            arguments[i] = \
              ((chunk_info_t*)((*(cl_mem *)\
                (al->value))->device_ptrs[0]))->start_address;
		//TODO: '0' above is the device number... don't hard-code!
        }
      else if (kernel->arg_is_image[i])
        {
          POCL_ABORT_UNIMPLEMENTED();
//          dev_image2d_t di;      
//          cl_mem mem = *(cl_mem*)al->value;
//          di.data = &((*(cl_mem *) (al->value))->device_ptrs[device]);
//          di.data = ((*(cl_mem *) (al->value))->device_ptrs[device]);
//          di.width = mem->image_width;
//          di.height = mem->image_height;
//          di.rowpitch = mem->image_row_pitch;
//          di.order = mem->image_channel_order;
//          di.data_type = mem->image_channel_data_type;
//          void* devptr = pocl_cellspu_malloc(data, 0, sizeof(dev_image2d_t), NULL);
//          arguments[i] = malloc (sizeof (void *));
//          *(void **)(arguments[i]) = devptr; 
//          pocl_cellspu_write (data, &di, devptr, sizeof(dev_image2d_t));
        }
      else if (kernel->arg_is_sampler[i])
        {
          POCL_ABORT_UNIMPLEMENTED();
//          dev_sampler_t ds;
//          
//          arguments[i] = malloc (sizeof (void *));
//          *(void **)(arguments[i]) = pocl_cellspu_malloc(data, 0, sizeof(dev_sampler_t), NULL);
//          pocl_cellspu_write (data, &ds, *(void**)arguments[i], sizeof(dev_sampler_t));
        }
      else
        {
          arguments[i] = (uint32_t)al->value;
        }
    }

  // allocate memory for kernel local variables
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      al = &(kernel->dyn_arguments[i]);
      arguments[i] = (uint32_t)malloc (sizeof (void *));
      *(void **)(arguments[i]) = cellspu_malloc_local(data, al->size);
    }

  // the main loop on the spe needs an auxiliary struct for to get the 
  // number of arguments and such. 
  __kernel_metadata kmd;
  strncpy( kmd.name, workgroup_string, sizeof( kmd.name ) );  
  kmd.num_args = kernel->num_args;
  kmd.num_locals = kernel->num_locals;
  // TODO: fill in the rest, if used by the spu main function.

  // TODO malloc_local should be given the 'device data'. as long as teh 
  // spu context is global this is ok.
  void *chunk = cellspu_malloc_local( NULL, sizeof(__kernel_metadata) ); 
  void *kernel_area = ((chunk_info_t*)chunk)->start_address;
  cellspu_memwrite( kernel_area, &kmd, sizeof(__kernel_metadata) );
  dev_cmd.kernel = kernel_area;
  
  // finish up the command, send it to SPE
  dev_cmd.status =POCL_KST_READY;
  cellspu_memwrite( (void*)CELLSPU_KERNEL_CMD_ADDR, &dev_cmd, sizeof(__kernel_exec_cmd) );
       
  // Execute code on SPU. This starts with the main() in the spu - see spe_wrap.c
  if (spe_context_run(spe_context,&entry,0,NULL,NULL,NULL) < 0)
    perror("context_run error");

//  for (z = 0; z < pc->num_groups[2]; ++z)
//    {
//      for (y = 0; y < pc->num_groups[1]; ++y)
//        {
//          for (x = 0; x < pc->num_groups[0]; ++x)
//            {
//              pc->group_id[0] = x;
//              pc->group_id[1] = y;
//              pc->group_id[2] = z;
//
//              w (arguments, pc);
//
//            }
//        }
//    }


  // Clean-up ? 
  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_is_local[i])
        pocl_cellspu_free(data, 0, *(void **)(arguments[i]));
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    pocl_cellspu_free(data, 0, *(void **)(arguments[i]));
}

void
pocl_cellspu_copy (void *data, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb)
{
  POCL_ABORT_UNIMPLEMENTED();

  if (src_ptr == dst_ptr)
    return;
  
  memcpy (dst_ptr, src_ptr, cb);
}

void
pocl_cellspu_copy_rect (void *data,
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
pocl_cellspu_write_rect (void *data,
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
  POCL_ABORT_UNIMPLEMENTED();

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              region[0]);
}

void
pocl_cellspu_read_rect (void *data,
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
  POCL_ABORT_UNIMPLEMENTED();
  
  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              region[0]);
}


void *
pocl_cellspu_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size,
                      void *host_ptr) 
{
  /* All global pointers of the pthread/CPU device are in 
     the host address space already, and up to date. */
  POCL_ABORT_UNIMPLEMENTED();

  if (host_ptr != NULL) return host_ptr;
  return buf_ptr + offset;
}

void
pocl_cellspu_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  POCL_ABORT_UNIMPLEMENTED();

  free (d);
  device->data = NULL;
}

cl_ulong
pocl_cellspu_get_timer_value (void *data) 
{
  POCL_ABORT_UNIMPLEMENTED();

  struct timeval current;
  gettimeofday(&current, NULL);  
  return (current.tv_sec * 1000000 + current.tv_usec)*1000;
}

int 
pocl_cellspu_build_program (void *data, char *source_fn, char *binary_fn, 
			    char *default_cmd, char *dev_tmpdir) 
{
  POCL_ABORT_UNIMPLEMENTED();

}

void *
pocl_cellspu_create_sub_buffer (void *device_data, void* buffer, size_t origin, size_t size)
{
  POCL_ABORT_UNIMPLEMENTED();
  return NULL;
}
