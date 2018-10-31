/* tce_common.cc - common functionality over the different TCE/TTA device drivers.

   Copyright (c) 2012-2014 Pekka Jääskeläinen / Tampere University of Technology
   
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
#include "tce_common.h"
#include "pocl_util.h"
#include "pocl_cache.h"
#include "pocl_llvm.h"
#include "utlist.h"
#include "common.h"

#include "config.h"
#include "pocl_runtime_config.h"
#include "pocl_hash.h"
#include "pocl_cache.h"

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

/* Supress some warnings because of including tce_config.h after pocl's config.h. */
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_VERSION
#undef VERSION
#undef SIZEOF_DOUBLE

#include <Machine.hh>
#include <Program.hh>
#include <DataLabel.hh>
#include <AddressSpace.hh>
#include <GlobalScope.hh>
#include <Environment.hh>

using namespace TTAMachine;

#include <algorithm>
#include <sstream>

#define ALIGNMENT MAX_EXTENDED_ALIGNMENT

//#define DEBUG_TTA_DRIVER

TCEDevice::TCEDevice(cl_device_id dev, const char* adfName) :
  local_as(NULL), global_as(NULL), private_as(NULL), machine_file(adfName), parent(dev),
  currentProgram(NULL), curKernelAddr(0), curKernel(NULL), globalCycleCount(0),
  ready_list(NULL), command_list(NULL) {
  parent->data = this;
  pthread_mutex_init (&cq_lock, NULL);
  POCL_INIT_LOCK(tce_compile_lock);
  dev->address_bits = 32;
  dev->autolocals_to_args = 1;
  /* This assumes TCE is always Little-endian;
   * needsByteSwap is set up again in TTASimDevice
   * after we know whether ADF is big- or little-endian. */
#if defined(WORDS_BIGENDIAN) && WORDS_BIGENDIAN == 1
  needsByteSwap = true;
#else
  needsByteSwap = false;
#endif
}

TCEDevice::~TCEDevice() {
  parent->data = NULL;
}

bool
TCEDevice::isMultiCoreMachine() const {
#ifdef TCEMC_AVAILABLE
  assert (machine_ != NULL);
  return machine_->coreCount() > 1;
#else
  return false;
#endif
}

/**
 * This should be called by the derived classes at the point the
 * TTA machine description is loaded. It loads additional device
 * properties from the parsed ADF.
 */
void
TCEDevice::setMachine(const TTAMachine::Machine& machine) {
  machine_ = &machine;
}

void
TCEDevice::writeWordToDevice(uint32_t dest_addr, uint32_t word) {
  uint32_t swapped = byteswap_uint32_t(word, needsByteSwap);
  copyHostToDevice(&swapped, dest_addr, sizeof (swapped));
}

uint32_t
TCEDevice::readWordFromDevice(uint32_t addr) {
  uint32_t result;
  copyDeviceToHost(addr, &result, sizeof(result));
  return byteswap_uint32_t(result, needsByteSwap);
}

void
TCEDevice::findDataMemoryAddresses() {
  /* Figure out the locations of the shared data structures in
     the device memories from the fully-linked program. 
  */
  const TTAProgram::Program* prog = currentProgram;
  assert (prog != NULL);

  const TTAProgram::GlobalScope& globalScope = prog->globalScopeConst();

  commandQueueAddr = global_as->start() + TTA_UNALLOCATED_GLOBAL_SPACE;
}

void
TCEDevice::initDataMemory() {
  findDataMemoryAddresses();
  writeWordToDevice(commandQueueAddr, POCL_KST_FREE); 
}

void
TCEDevice::initMemoryManagement(const TTAMachine::Machine& mach) {
  /* Create the memory allocation book keeping structures based on
     the machine's address spaces (see tta.txt). */
  Machine::AddressSpaceNavigator nav = mach.addressSpaceNavigator();

  for (int i = 0; i < nav.count(); ++i) {
    AddressSpace *as = nav.item(i);
    if (as->hasNumericalId(TTA_ASID_LOCAL)) {
      local_as = as;
    } 
    if (as->hasNumericalId(TTA_ASID_PRIVATE)) {
      private_as = as;
    }
    if (as->hasNumericalId(TTA_ASID_GLOBAL) &&
        as->hasNumericalId(TTA_ASID_CONSTANT)) {
      global_as = as;
    }
  }
  if (local_as == NULL) 
    POCL_ABORT("local address space not found in the ADF. "
               "Mark it by adding numerical id 4 to the AS.\n"
	       "Local address space can be same as private AS.\n");


  if (isMultiCoreMachine() && local_as->isShared()) 
    POCL_ABORT("The local address space is marked as shared!\n");

  if (private_as == NULL) 
    POCL_ABORT("private address space not found in the ADF. "
               "Mark it by adding numerical id 0 to the AS.\n"
	       "Private address space can be same as local AS.\n");

  if (isMultiCoreMachine() && private_as->isShared()) 
    POCL_ABORT("The private address space is marked as shared!\n");

  if (global_as == NULL) 
    POCL_ABORT("global address space not found in the ADF. "
               "Mark it by adding numerical ids 3 and 5 to the AS.\n");

  if (isMultiCoreMachine() && !global_as->isShared()) 
    POCL_ABORT("The global address space is not marked as shared!\n");

  int local_size = (private_as == local_as) ?
    local_as->end() - local_as->start() - TTA_UNALLOCATED_LOCAL_SPACE:
    local_as->end() - local_as->start();
  if (local_size < 0)
    POCL_ABORT("Not enough space in the local memory with the assumed unallocated space.\n");

  parent->local_mem_size = local_size;
  int global_size = global_as->end() - local_as->start() - TTA_UNALLOCATED_GLOBAL_SPACE;
  if (global_size < 0)
    POCL_ABORT("Not enough space in the global memory with the assumed unallocated space.\n");
  parent->global_mem_size = global_size;
  parent->max_mem_alloc_size = global_size;

  init_mem_region 
    (&local_mem, (memory_address_t)local_as->start(), parent->local_mem_size);
  init_mem_region 
    (&global_mem, (memory_address_t)global_as->start() + TTA_UNALLOCATED_GLOBAL_SPACE + sizeof(__kernel_exec_cmd),
     parent->global_mem_size);
}

#define SUBST(x) "  -DKERNEL_EXE_CMD_OFFSET=" # x
#define OFFSET_ARG(c) SUBST(c)

TCEString TCEDevice::tceccCommandLine(_cl_command_run *run_cmd,
                                      const TCEString &tempDir,
                                      const TCEString &inputSrc,
                                      const TCEString &outputTpef,
                                      const TCEString extraParams) {

  TCEString mainC;
  if (isMultiCoreMachine()) 
    mainC = "tta_device_main_dthread.c";
  else
    mainC = "tta_device_main.c";

  TCEString deviceMainSrc;
  TCEString poclIncludePathSwitch;
  if (pocl_get_bool_option("POCL_BUILDING", 0))
    {
      deviceMainSrc = TCEString(SRCDIR) + "/lib/CL/devices/tce/" + mainC;
      poclIncludePathSwitch = " -I " SRCDIR "/include";
    }
  else 
    {
      deviceMainSrc = TCEString(POCL_INSTALL_PRIVATE_DATADIR) + "/" + mainC;
      assert(access(deviceMainSrc.c_str(), R_OK) == 0);
      poclIncludePathSwitch = " -I " POCL_INSTALL_PRIVATE_DATADIR "/include";
    }

  TCEString extraFlags = extraParams;
  if (isMultiCoreMachine())
    extraFlags += " -ldthread -lsync-lu -llockunit";

  extraFlags += OFFSET_ARG(TTA_UNALLOCATED_GLOBAL_SPACE);

  std::string kernelObjSrc = "";
  kernelObjSrc += tempDir;
  kernelObjSrc += "/../descriptor.so.kernel_obj.c";

  if (pocl_is_option_set("POCL_TCECC_EXTRA_FLAGS"))
    extraFlags += " " + 
      TCEString(pocl_get_string_option("POCL_TCECC_EXTRA_FLAGS", ""));
  if (parent->endian_little) {
    extraFlags += " --little-endian";
  }

  std::string kernelMdSymbolName = "_";
  kernelMdSymbolName += run_cmd->kernel->name;
  kernelMdSymbolName += "_md";

  TCEString programBcFile = tempDir + "/program.bc";
  /* Compile in steps to save the program.bc for automated exploration 
     use case when producing the kernel capture scripts. */
  TCEString cmdLine;
  cmdLine << "tcecc -llwpr " + poclIncludePathSwitch + " " + deviceMainSrc + " " + 
    " " + kernelObjSrc + " " + inputSrc +
    " -k " + kernelMdSymbolName +
    " -g -O3 --emit-llvm -o " + programBcFile + " " + extraFlags + ";";

  cmdLine << "tcecc $* -a " << machine_file << " " << programBcFile 
          << " -O3 -o " << outputTpef << + " " + extraFlags + "\n";
  return cmdLine;
}

bool 
TCEDevice::isNewKernel(const _cl_command_run* runCmd) 
{
  if (curKernel == NULL || runCmd->kernel != curKernel) 
    return true;

  bool newKernel = true;
  if (runCmd->local_x != curLocalX ||
      runCmd->local_y != curLocalY ||
      runCmd->local_z != curLocalZ)
    newKernel = true;
  else
    newKernel = false;
  return newKernel;
}


void 
TCEDevice::updateCurrentKernel(const _cl_command_run* runCmd, 
                               uint32_t kernelAddr)
{
  curKernelAddr = kernelAddr;
  curKernel = runCmd->kernel;
  curLocalX = runCmd->local_x;
  curLocalY = runCmd->local_y;
  curLocalZ = runCmd->local_z;
}

void *
pocl_tce_malloc (void *device_data, cl_mem_flags flags,
                 size_t size, void *host_ptr)
{
  TCEDevice *d = (TCEDevice*)device_data;

  chunk_info_t *chunk = alloc_buffer (&d->global_mem, size);
  if (chunk == NULL) return NULL;

#ifdef DEBUG_TTA_DRIVER
  printf("host: malloc %p : %lu / %zu\n", host_ptr, chunk->start_address, size);
#endif

  if ((flags & CL_MEM_COPY_HOST_PTR) ||  
      ((flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL))
    {
      /* TODO: 
         CL_MEM_USE_HOST_PTR must synch the buffer after execution 
         back to the host's memory in case it's used as an output (?). */
      d->copyHostToDevice(host_ptr, chunk->start_address, size);
      return (void*) chunk;
    }
  return (void*) chunk;
}

cl_int
pocl_tce_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, void* host_ptr)
{
  void *b = NULL;
  cl_int flags = mem_obj->flags;
  unsigned i;

  /* check if some driver has already allocated memory for this mem_obj 
     in our global address space, and use that*/
  for (i = 0; i < mem_obj->context->num_devices; ++i)
    {
      if (!mem_obj->device_ptrs[i].available)
        continue;

      if (mem_obj->device_ptrs[i].global_mem_id == device->global_mem_id
          && mem_obj->device_ptrs[i].mem_ptr != NULL)
        {
          mem_obj->device_ptrs[device->dev_id].mem_ptr =
            mem_obj->device_ptrs[i].mem_ptr;

          return CL_SUCCESS;
        }
    }

  b = pocl_tce_malloc
    (device->data, flags, mem_obj->size, host_ptr);
  if (b == NULL) return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  mem_obj->device_ptrs[device->dev_id].mem_ptr = b;

  return CL_SUCCESS;

}

void
pocl_tce_write (void *data,
                const void *__restrict__  src_host_ptr,
                pocl_mem_identifier * dst_mem_id,
                cl_mem dst_buf,
                size_t offset, size_t size)
{
  void *__restrict__ device_ptr = dst_mem_id->mem_ptr;
  TCEDevice *d = (TCEDevice*)data;
  chunk_info_t *chunk = (chunk_info_t*)device_ptr;
#ifdef DEBUG_TTA_DRIVER
  printf ("host: write %p <- %lx / %zu\n", src_host_ptr, chunk->start_address + offset,
          size);
#endif
  d->copyHostToDevice (src_host_ptr, chunk->start_address + offset, size);
}

void
pocl_tce_read (void *data,
               void *__restrict__ dst_host_ptr,
               pocl_mem_identifier * src_mem_id,
               cl_mem src_buf,
               size_t offset,
               size_t size)
{
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;
  TCEDevice* d = (TCEDevice*)data;
  chunk_info_t *chunk = (chunk_info_t*)device_ptr;
#ifdef DEBUG_TTA_DRIVER
  printf ("host: read %p -> %lx / %zu\n", dst_host_ptr,
          chunk->start_address + offset, size);
#endif
  d->copyDeviceToHost (chunk->start_address + offset, dst_host_ptr, size);
}

chunk_info_t*
pocl_tce_malloc_local (void *device_data, size_t size) 
{
  TCEDevice *d = (TCEDevice*)device_data;
  return alloc_buffer (&d->local_mem, size);
}

void
pocl_tce_free (cl_device_id device, cl_mem mem_obj)
{
  void* ptr = mem_obj->device_ptrs[device->dev_id].mem_ptr;
  free_chunk ((chunk_info_t*) ptr);
}

static void pocl_tce_write_kernel_descriptor(cl_device_id device,
                                             unsigned device_i,
                                             cl_kernel kernel) {
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous standalone devices which
  // need the definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now

  std::stringstream content;
  pocl_kernel_metadata_t *meta = kernel->meta;

  content << std::endl
          << "#include <pocl_device.h>" << std::endl
          << "void _pocl_launcher_" << meta->name
          << "_workgroup(uint8_t* args, uint8_t*, "
          << "uint32_t, uint32_t, uint32_t);" << std::endl
          << "void _pocl_launcher_" << meta->name
          << "_workgroup_fast(uint8_t* args, uint8_t*, "
          << "uint32_t, uint32_t, uint32_t);" << std::endl;

  if (device->global_as_id != 0)
    content << "__attribute__((address_space(" << device->global_as_id << ")))"
            << std::endl;

  content << "__kernel_metadata _" << meta->name << "_md = {" << std::endl
          << "     \"" << meta->name << "\"," << std::endl
          << "     " << meta->num_args << "," << std::endl
          << "     " << meta->num_locals << "," << std::endl
          << "     _pocl_launcher_" << meta->name << "_workgroup_fast"
          << std::endl
          << " };" << std::endl;

  pocl_cache_write_descriptor(kernel->program, device_i, meta->name,
                              content.str().c_str(), content.str().size());
}

void
pocl_tce_compile_kernel(_cl_command_node *cmd,
                        cl_kernel kernel, cl_device_id device)
{
  if (cmd->type != CL_COMMAND_NDRANGE_KERNEL)
    return;

  void* data = cmd->device->data;
  TCEDevice *d = (TCEDevice*)data;

  if (!kernel)
    kernel = cmd->command.run.kernel;
  if (!device)
    device = cmd->device;

  POCL_LOCK(d->tce_compile_lock);
  int error = pocl_llvm_generate_workgroup_function(
      cmd->command.run.device_i, device, kernel, cmd->command.run.local_x,
      cmd->command.run.local_y, cmd->command.run.local_z);

  if (error) {
    POCL_UNLOCK(d->tce_compile_lock);
    POCL_MSG_PRINT_GENERAL("TCE: pocl_llvm_generate_workgroup_function()"
                           " failed for kernel %s\n", kernel->name);
    assert(error == 0);
  }

  // 12 == strlen (POCL_PARALLEL_BC_FILENAME)
  char bytecode[POCL_FILENAME_LENGTH + 13];

  assert(d != NULL);
  assert(cmd->command.run.kernel);

  char cachedir[POCL_FILENAME_LENGTH];
  pocl_cache_kernel_cachedir_path(cachedir, kernel->program,
                                  cmd->command.run.device_i, kernel, "",
                                  cmd->command.run.local_x,
                                  cmd->command.run.local_y,
                                  cmd->command.run.local_z);
  cmd->command.run.device_data = strdup(cachedir);

  if (d->isNewKernel(&(cmd->command.run))) {
    pocl_tce_write_kernel_descriptor(device, cmd->command.run.device_i, kernel);

    std::string assemblyFileName(cachedir);
    TCEString tempDir(cachedir);
    assemblyFileName += "/parallel.tpef";

    if (access (assemblyFileName.c_str(), F_OK) != 0)
      {
      error = snprintf(bytecode, POCL_FILENAME_LENGTH, "%s%s", cachedir,
                       POCL_PARALLEL_BC_FILENAME);
      TCEString buildCmd = d->tceccCommandLine(&cmd->command.run, tempDir,
                                               bytecode, assemblyFileName);

#ifdef DEBUG_TTA_DRIVER
      std::cerr << "CMD: " << buildCmd << std::endl;
#endif
      error = system(buildCmd.c_str());
      if (error != 0)
        POCL_ABORT("Error while running tcecc.\n");
      }
  }

  POCL_UNLOCK(d->tce_compile_lock);
}

void
pocl_tce_run(void *data, _cl_command_node* cmd)
{
  assert(cmd->type == CL_COMMAND_NDRANGE_KERNEL);

  TCEDevice *d = (TCEDevice*)data;
  uint32_t kernelAddr;
  unsigned i;

  assert(d != NULL);
  assert(cmd->command.run.kernel);
  assert(cmd->command.run.device_data);

  if (d->isNewKernel(&(cmd->command.run))) {
    std::string assemblyFileName((const char*)cmd->command.run.device_data);
    assemblyFileName += "/parallel.tpef";

    std::string kernelMdSymbolName = "_";
    kernelMdSymbolName += cmd->command.run.kernel->name;
    kernelMdSymbolName += "_md";

    try {
      d->loadProgramToDevice(assemblyFileName);
      d->restartProgram();
    } catch (Exception &e) {
      std::cerr << "error: " << e.errorMessage() << std::endl;
      POCL_ABORT("error: Failed to load program to the TTA.\n");
    }

    const TTAProgram::Program* prog = d->currentProgram;
    assert (prog != NULL);
    
    const TTAProgram::GlobalScope& globalScope = prog->globalScopeConst();
    
    try {
      kernelAddr = globalScope.dataLabel(kernelMdSymbolName).address().location();
    } catch (const KeyNotFound& e) {
      POCL_ABORT("Could not find the shared data structures from the device binary.\n");
    }
    // cache the currently device loaded kernel info 
    d->updateCurrentKernel(&(cmd->command.run), kernelAddr);
  } else {
    // Same kernel, no need to recompile
    d->restartProgram();
    kernelAddr = d->curKernelAddr;
  }
  __kernel_exec_cmd dev_cmd;
  dev_cmd.kernel = byteswap_uint32_t (kernelAddr, d->needsByteSwap);

  struct pocl_argument *al;  

  typedef std::vector<chunk_info_t*> ChunkVector;
  /* Chunks to be freed after the kernel finishes. */
  ChunkVector tempChunks;

  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;

  for (i = 0; i < meta->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          chunk_info_t* local_chunk = pocl_tce_malloc_local (d, al->size);
          if (local_chunk == NULL)
            POCL_ABORT ("Could not allocate memory for a local argument. Out of local mem?\n");

          dev_cmd.args[i] = byteswap_uint32_t (local_chunk->start_address, d->needsByteSwap);
#ifdef DEBUG_TTA_DRIVER
          printf ("host: allocated %zu bytes of local memory for arg %u @ %lu\n",
                  al->size, i, local_chunk->start_address);
#endif
          tempChunks.push_back(local_chunk);
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            dev_cmd.args[i] = 0;
          else
            {
              cl_mem m = (*(cl_mem *)(al->value));
              void *p = m->device_ptrs[d->parent->dev_id].mem_ptr;
              dev_cmd.args[i] = byteswap_uint32_t (
                  ((chunk_info_t *)p)->start_address + al->offset,
                  d->needsByteSwap);
            }
        }
      else /* The scalar values should be byteswapped by the user. */
        {
          /* Copy the scalar argument data to the shared memory. */
          chunk_info_t* arg_space = 
            (chunk_info_t*)pocl_tce_malloc (d, CL_MEM_COPY_HOST_PTR, al->size, al->value);
          if (arg_space == NULL)
            POCL_ABORT ("Could not allocate memory from the device argument space. Out of global mem?\n");
#ifdef DEBUG_TTA_DRIVER
          printf ("host: copied value from %p to global argument memory\n", al->value);
#endif
          dev_cmd.args[i] = byteswap_uint32_t (arg_space->start_address, d->needsByteSwap);
          tempChunks.push_back(arg_space);
        }
    }

  /* Allocate the automatic local buffers. */
  for (i = 0; i < meta->num_locals; ++i)
    {
      size_t s = meta->local_sizes[i];
      chunk_info_t* local_chunk = pocl_tce_malloc_local (d, s);
      if (local_chunk == NULL)
        POCL_ABORT ("Could not allocate memory for an automatic local argument. Out of local mem?\n");

      dev_cmd.args[meta->num_args + i] = byteswap_uint32_t (local_chunk->start_address, d->needsByteSwap);
#ifdef DEBUG_TTA_DRIVER
      printf ("host: allocated %zu bytes of local memory for automated local arg %u @ %lu\n",
              s, (meta->num_args + i), local_chunk->start_address);
#endif      
      tempChunks.push_back(local_chunk);
    }
  dev_cmd.work_dim = byteswap_uint32_t (cmd->command.run.pc.work_dim, d->needsByteSwap);
  dev_cmd.num_groups[0] = byteswap_uint32_t (cmd->command.run.pc.num_groups[0], d->needsByteSwap);
  dev_cmd.num_groups[1] = byteswap_uint32_t (cmd->command.run.pc.num_groups[1], d->needsByteSwap);
  dev_cmd.num_groups[2] = byteswap_uint32_t (cmd->command.run.pc.num_groups[2], d->needsByteSwap);

  dev_cmd.global_offset[0] = byteswap_uint32_t (cmd->command.run.pc.global_offset[0], d->needsByteSwap);
  dev_cmd.global_offset[1] = byteswap_uint32_t (cmd->command.run.pc.global_offset[1], d->needsByteSwap);
  dev_cmd.global_offset[2] = byteswap_uint32_t (cmd->command.run.pc.global_offset[2], d->needsByteSwap);

  dev_cmd.status = byteswap_uint32_t (POCL_KST_FREE, d->needsByteSwap);

#ifdef DEBUG_TTA_DRIVER
  printf("host: waiting for the device command queue (@ %x) to get room.\n",
         d->commandQueueAddr);
  printf("host: command queue status: %d\n",
         d->readWordFromDevice (d->commandQueueAddr));
#endif
  /* Wait until the device command queue has room. */
  do {} 
  while (d->readWordFromDevice (d->commandQueueAddr) != POCL_KST_FREE);

#ifdef DEBUG_TTA_DRIVER
  printf( "host: writing the command.\n");
#endif
  d->copyHostToDevice (&dev_cmd, d->commandQueueAddr, sizeof(__kernel_exec_cmd) );

  /* Ensure the READY status is written the last so the device doesn't
     start executing before all the cmd data has been written. We 
     need a flush or similar mechanism to ensure all the data has 
     been really written, in case the data transfers are not guaranteed
     to be ordered. */
  d->writeWordToDevice(d->commandQueueAddr, POCL_KST_READY);

  dev_cmd.status = byteswap_uint32_t (POCL_KST_READY, d->needsByteSwap);

  d->notifyKernelRunCommandSent(dev_cmd, &cmd->command.run);

#ifdef DEBUG_TTA_DRIVER
  printf("host: commmand queue status: %x\n",
         d->readWordFromDevice(d->commandQueueAddr));

  printf("host: waiting for the command to get executed.\n");
#endif
  /* Wait until the command has executed. */
  do {
#ifdef DEBUG_TTA_DRIVER
      printf("host: commmand queue status: %x\n",
             d->readWordFromDevice(d->commandQueueAddr));
      sleep(1);
#endif
  usleep(20000);
  } while (d->readWordFromDevice(d->commandQueueAddr) != POCL_KST_FINISHED);

#ifdef DEBUG_TTA_DRIVER
  printf( "host: done. Freeing the command queue entry.\n");
#endif
  /* We are done with this kernel, free the command queue entry. */
  d->writeWordToDevice(d->commandQueueAddr, POCL_KST_FREE);

  for (ChunkVector::iterator i = tempChunks.begin(); 
       i != tempChunks.end(); ++i) 
    free_chunk (*i);

  POCL_MEM_FREE(cmd->command.run.device_data);

#ifdef DEBUG_TTA_DRIVER
  printf("host: local memory allocations:\n");
  print_chunks (d->local_mem.chunks);

  printf("host: global memory allocations:\n");
  print_chunks (d->global_mem.chunks);
#endif  
}

cl_int
pocl_tce_map_mem (void *data,
                  pocl_mem_identifier * src_mem_id,
                  cl_mem src_buf,
                  mem_mapping_t *map)
{
  if (map->host_ptr == NULL)
    {
      map->host_ptr = pocl_aligned_malloc (ALIGNMENT, map->size);
      return CL_SUCCESS;
    }

  /* Synch the device global region to the host memory. */
  pocl_tce_read (data, map->host_ptr, src_mem_id, src_buf, map->offset, map->size);
  return CL_SUCCESS;
}

cl_int
pocl_tce_unmap_mem (void *data,
                    pocl_mem_identifier *dst_mem_id,
                    cl_mem dst_buf,
                    mem_mapping_t *map)
{
  if (map->map_flags != CL_MAP_READ)
    /* Synch the device global region to the host memory. */
    pocl_tce_write (data, map->host_ptr, dst_mem_id, dst_buf, map->offset, map->size);

  if (map->host_ptr != ((char *)dst_buf->mem_host_ptr + map->offset))
    {
      pocl_aligned_free (map->host_ptr);
    }

  return CL_SUCCESS;
}


char* 
pocl_tce_init_build(void *data)
{
  TCEDevice *tce_dev = (TCEDevice*)data;
  TCEString mach_tmpdir =
      Environment::llvmtceCachePath();

  TCEString mach_header_base =
      mach_tmpdir + "/" + tce_dev->machine_->hash();

  int error = 0;

  std::string devextHeaderFn =
    std::string(mach_header_base) + std::string("_opencl_devext.h");

  /* Generate the vendor extensions header to provide explicit
     access to the (custom) hardware operations. */
  std::string tceopgenCmd =
      std::string("tceopgen > ") + devextHeaderFn;

  error = system (tceopgenCmd.c_str());
  if (error == -1) return NULL;

  std::string extgenCmd = 
    std::string("tceoclextgen ") + tce_dev->machine_file + 
      std::string(" >> ") + devextHeaderFn;

  error = system (extgenCmd.c_str());
  if (error == -1) return NULL;

  // gnu-keywords needed to support the inline asm blocks
  // -fasm doesn't work in the frontend

  std::string includeSwitch = 
    std::string("-fgnu-keywords -Dasm=__asm__ -include ") + devextHeaderFn;
  
  char *include_switch = strdup(includeSwitch.c_str());

  return include_switch;
}

char *
pocl_tce_build_hash (cl_device_id device)
{
  TCEDevice *tce_dev = (TCEDevice*)device->data;
  FILE* adf_file = fopen (tce_dev->machine_file.c_str(), "r");
  size_t size;
  uint8_t* adf_data = 0;
  const char *extra_flags = NULL;

  fseek (adf_file, 0 , SEEK_END);
  size = ftell (adf_file);
  fseek (adf_file, 0, SEEK_SET);
  adf_data = (uint8_t*)malloc (size);
  if (fread (adf_data, 1, size, adf_file) == 0)
      POCL_ABORT("Could not read ADF.\n");

  SHA1_CTX ctx;
  uint8_t bin_dig[SHA1_DIGEST_SIZE];
  pocl_SHA1_Init(&ctx);
  pocl_SHA1_Update(&ctx, adf_data, size);
  pocl_SHA1_Final(&ctx, bin_dig);

  char *result = (char *)calloc(1000, sizeof(char));
  strcpy(result, device->llvm_target_triplet);
  char *temp = result + strlen(result);
  *temp++ = '-';
  unsigned i;
  for (i=0; i < SHA1_DIGEST_SIZE; i++)
    {
      *temp++ = (bin_dig[i] & 0x0F) + 65;
      *temp++ = ((bin_dig[i] & 0xF0) >> 4) + 65;
    }
  *temp++ = '_';
  *temp = 0;

  if (pocl_is_option_set("POCL_TCECC_EXTRA_FLAGS"))
    {
      extra_flags = pocl_get_string_option("POCL_TCECC_EXTRA_FLAGS", "");
      strncpy(temp, extra_flags, (1000-(temp-result)) );
    }

  return result;
}

void
pocl_tce_copy (void */*data*/,
               pocl_mem_identifier * dst_mem_id,
               cl_mem dst_buf,
               pocl_mem_identifier * src_mem_id,
               cl_mem src_buf,
               size_t dst_offset,
               size_t src_offset,
               size_t size)
{
  POCL_ABORT_UNIMPLEMENTED("Copy not yet supported in TCE driver.");
  if (src_mem_id->mem_ptr == dst_mem_id->mem_ptr)
    return;

  memcpy (dst_mem_id->mem_ptr, src_mem_id->mem_ptr, size);
}

void
pocl_tce_copy_rect (void *data,
                    pocl_mem_identifier * dst_mem_id,
                    cl_mem dst_buf,
                    pocl_mem_identifier * src_mem_id,
                    cl_mem src_buf,
                    const size_t *__restrict__ const dst_origin,
                    const size_t *__restrict__ const src_origin,
                    const size_t *__restrict__ const region,
                    size_t const dst_row_pitch,
                    size_t const dst_slice_pitch,
                    size_t const src_row_pitch,
                    size_t const src_slice_pitch)
{
  TCEDevice *d = (TCEDevice*)data;
  chunk_info_t *src_chunk = (chunk_info_t*)src_mem_id->mem_ptr;
  chunk_info_t *dst_chunk = (chunk_info_t*)dst_mem_id->mem_ptr;

  size_t src_offset = src_origin[0] + src_row_pitch * (src_origin[1] + src_slice_pitch * src_origin[2]);
  size_t dst_offset = dst_origin[0] + dst_row_pitch * (dst_origin[1] + dst_slice_pitch * dst_origin[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      d->copyDeviceToDevice(src_chunk->start_address + src_offset + src_row_pitch * j + src_slice_pitch * k,
                            dst_chunk->start_address + dst_offset + dst_row_pitch * j + dst_slice_pitch * k,
                            region[0]);

}

void
pocl_tce_write_rect (void *data,
                     const void *__restrict__ src_host_ptr,
                     pocl_mem_identifier * dst_mem_id,
                     cl_mem dst_buf,
                     const size_t *__restrict__ const buffer_origin,
                     const size_t *__restrict__ const host_origin, 
                     const size_t *__restrict__ const region,
                     size_t const buffer_row_pitch,
                     size_t const buffer_slice_pitch,
                     size_t const host_row_pitch,
                     size_t const host_slice_pitch)
{
  const void *__restrict__ const host_ptr = src_host_ptr;
  void *__restrict__ const device_ptr = dst_mem_id->mem_ptr;

  char const *__restrict__ const adjusted_host_ptr =
    (char const*)host_ptr +
    host_origin[0] + host_row_pitch * host_origin[1] + 
    host_slice_pitch * host_origin[2];
  
  size_t j, k;
  size_t base_offset = buffer_origin[0] + buffer_row_pitch * buffer_origin[1] 
    + buffer_slice_pitch * buffer_origin[2];
  /* TODO: handle overlaping regions */
    
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      {
        char const *__restrict h_ptr = adjusted_host_ptr + host_row_pitch * j 
          + host_slice_pitch * k;       
        size_t offset = base_offset + buffer_row_pitch * j 
          + buffer_slice_pitch * k;
        pocl_tce_write (data, h_ptr, dst_mem_id, dst_buf, offset, region[0]);
  
      }
}

void
pocl_tce_read_rect (void *data,
                    void *__restrict__ dst_host_ptr,
                    pocl_mem_identifier * src_mem_id,
                    cl_mem src_buf,
                    const size_t *__restrict__ const buffer_origin,
                    const size_t *__restrict__ const host_origin, 
                    const size_t *__restrict__ const region,
                    size_t const buffer_row_pitch,
                    size_t const buffer_slice_pitch,
                    size_t const host_row_pitch,
                    size_t const host_slice_pitch)
{
  void *__restrict__ const host_ptr = dst_host_ptr;
  void *__restrict__ const device_ptr = src_mem_id->mem_ptr;

  char *__restrict__ const adjusted_host_ptr =
    (char*)host_ptr +
    host_origin[0] + host_row_pitch * host_origin[1] + 
    host_slice_pitch * host_origin[2];
  
  size_t j, k;
  size_t base_offset = buffer_origin[0] + buffer_row_pitch * buffer_origin[1] 
    + buffer_slice_pitch * buffer_origin[2];
  /* TODO: handle overlaping regions */ 
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      {
        char *__restrict__ h_ptr = adjusted_host_ptr + host_row_pitch * j 
          + host_slice_pitch * k;       
        size_t offset = base_offset + buffer_row_pitch * j 
          + buffer_slice_pitch * k;
        pocl_tce_read (data, h_ptr, src_mem_id, src_buf, offset, region[0]);
      }
}

static void tce_command_scheduler (TCEDevice *d) 
{
  _cl_command_node *node;
  
  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready(node->event));
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK(d->cq_lock);
      assert (node->event->status == CL_SUBMITTED);
      if (node->type == CL_COMMAND_NDRANGE_KERNEL)
        pocl_tce_compile_kernel(node, NULL, NULL);
      pocl_exec_command(node);
      POCL_LOCK(d->cq_lock);
    }
    
  return;
}

void
pocl_tce_submit (_cl_command_node *node, cl_command_queue /*cq*/)
{
  TCEDevice *d = (TCEDevice*)node->device->data;

  node->ready = 1;
  POCL_LOCK(d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);
  POCL_UNLOCK_OBJ(node->event);

  tce_command_scheduler (d);
  POCL_UNLOCK(d->cq_lock);

  return;
}

void pocl_tce_flush (cl_device_id device, cl_command_queue /*cq*/)
{
  TCEDevice *d = (TCEDevice*)device->data;

  POCL_LOCK (d->cq_lock);
  tce_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}


void
pocl_tce_join(cl_device_id device, cl_command_queue /*cq*/)
{
  TCEDevice *d = (TCEDevice*)device->data;

  POCL_LOCK (d->cq_lock);
  tce_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_tce_notify (cl_device_id device, cl_event event, cl_event finished)
{
  TCEDevice *d = (TCEDevice*)device->data;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE) {
    POCL_UPDATE_EVENT_FAILED(event);
    return;
  }

  if (!node->ready)
    return;

  if (pocl_command_is_ready(event)) {
    if (event->status == CL_QUEUED) {
      POCL_UPDATE_EVENT_SUBMITTED(event);
      POCL_LOCK(d->cq_lock);
      CDL_DELETE(d->command_list, node);
      CDL_PREPEND(d->ready_list, node);
      POCL_UNLOCK(d->cq_lock);
    }
  }
}

void
pocl_tce_broadcast (cl_event event)
{
  pocl_broadcast (event);
}
