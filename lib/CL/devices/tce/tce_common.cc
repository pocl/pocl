/* tce_common.cc - common functionality over the different TCE/TTA device
   drivers.

   Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University of Technology
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "common.h"
#include "pocl_cache.h"
#include "pocl_device.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "tce_common.h"
#include "utlist.h"

#include "common_driver.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cache.h"

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

/* Supress some warnings because of including tce_config.h after pocl's config.h. */
#undef LLVM_VERSION

#include <Machine.hh>
#include <Program.hh>
#include <DataLabel.hh>
#include <AddressSpace.hh>
#include <GlobalScope.hh>
#include <Environment.hh>

using namespace TTAMachine;

#include <algorithm>
#include <random>
#include <sstream>

#define ALIGNMENT MAX_EXTENDED_ALIGNMENT

//#define DEBUG_TTA_DRIVER

typedef struct pocl_tce_event_data_s {
  pocl_cond_t event_cond;
} pocl_tce_event_data_t;

typedef struct tce_queue_data_s {
  pocl_cond_t cq_cond;
} tce_queue_data_t;

TCEDevice::TCEDevice(cl_device_id dev, const char *adfName)
    : local_as(NULL), global_as(NULL), private_as(NULL), machine_file(adfName),
      parent(dev), currentProgram(NULL), curKernelAddr(0), globalCycleCount(0),
      available(CL_TRUE), curKernel(NULL), shutdownRequested(false),
      work_queue(NULL) {

  POCL_INIT_LOCK(wq_lock);
  POCL_INIT_COND(wakeup_cond);
  POCL_INIT_LOCK(tce_compile_lock);
  dev->address_bits = 32;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
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
  POCL_DESTROY_LOCK(wq_lock);
  POCL_DESTROY_COND(wakeup_cond);
  POCL_DESTROY_LOCK(tce_compile_lock);
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
  uint32_t swapped = pocl_byteswap_uint32_t(word, needsByteSwap);
  copyHostToDevice(&swapped, dest_addr, sizeof (swapped));
}

uint32_t
TCEDevice::readWordFromDevice(uint32_t addr) {
  uint32_t result = 0;
  copyDeviceToHost(addr, &result, sizeof(result));
  return pocl_byteswap_uint32_t(result, needsByteSwap);
}

void
TCEDevice::findDataMemoryAddresses() {
  /* Figure out the locations of the shared data structures in
     the device memories from the fully-linked program. */
  const TTAProgram::Program* prog = currentProgram;
  assert (prog != NULL);
  commandQueueAddr = global_as->start() + TTA_UNALLOCATED_GLOBAL_SPACE;
  statusAddr = commandQueueAddr + offsetof(__kernel_exec_cmd, status);
}

void
TCEDevice::initDataMemory() {
  findDataMemoryAddresses();
  writeWordToDevice(statusAddr, POCL_KST_FREE);
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

  pocl_init_mem_region(&local_mem,
                       (memory_address_t)local_as->end() - local_size,
                       parent->local_mem_size);
  pocl_init_mem_region
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
  cmdLine << OACC_EXECUTABLE << " -llwpr " + poclIncludePathSwitch + " " + deviceMainSrc +
                 " " + " " + kernelObjSrc + " " + inputSrc + " -k " +
                 kernelMdSymbolName + " -g -O2 --emit-llvm -o " +
                 programBcFile + " " + extraFlags + ";";

  cmdLine << OACC_EXECUTABLE << " -a " << machine_file << " " << programBcFile << " -O1 -o "
          << outputTpef << +" " + extraFlags + "\n";
  return cmdLine;
}

bool TCEDevice::isNewKernel(const _cl_command_run *runCmd) {
  if (curKernel == NULL || runCmd->kernel != curKernel)
    return true;

  bool newKernel = true;
  if (runCmd->pc.local_size[0] != curLocalX ||
      runCmd->pc.local_size[1] != curLocalY ||
      runCmd->pc.local_size[2] != curLocalZ ||
      runCmd->pc.global_offset[0] != curGoffsX ||
      runCmd->pc.global_offset[1] != curGoffsY ||
      runCmd->pc.global_offset[2] != curGoffsZ)
    newKernel = true;
  else
    newKernel = false;
  return newKernel;
}

void TCEDevice::updateCurrentKernel(const _cl_command_run *runCmd,
                                    uint32_t kernelAddr) {
  curKernelAddr = kernelAddr;
  curKernel = runCmd->kernel;
  curLocalX = runCmd->pc.local_size[0];
  curLocalY = runCmd->pc.local_size[1];
  curLocalZ = runCmd->pc.local_size[2];
  curGoffsX = runCmd->pc.global_offset[0];
  curGoffsY = runCmd->pc.global_offset[1];
  curGoffsZ = runCmd->pc.global_offset[2];
}

cl_int
pocl_tce_alloc_mem_obj (cl_device_id device, cl_mem mem, void* host_ptr)
{
  TCEDevice *d = (TCEDevice*)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  assert (p->mem_ptr == NULL);
  chunk_info_t *chunk = NULL;
  int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  /* TCE driver doesn't preallocate */
  if ((mem->flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == nullptr))
    goto ERROR;

  chunk = pocl_alloc_buffer_from_region(&d->global_mem, mem->size);
  if (chunk == NULL)
    goto ERROR;

  POCL_MSG_PRINT_MEMORY("TCE: ALLOC %zu bytes START AT %zu\n", mem->size,
                        chunk->start_address);

  p->mem_ptr = chunk;
  p->version = 0;
  err = CL_SUCCESS;

ERROR:
  return err;
}

void
pocl_tce_free (cl_device_id device, cl_mem mem) {

  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  assert (p->mem_ptr != NULL);

  chunk_info_t *chunk =
      (chunk_info_t *)p->mem_ptr;

  POCL_MSG_PRINT_MEMORY("TCE: FREE 0x%zu bytes START AT 0x%zu\n", mem->size,
                        chunk->start_address);

  pocl_free_chunk(chunk);

  p->mem_ptr = NULL;
  p->version = 0;
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
  POCL_MSG_PRINT_TCE("WRITE %p -> %u + %u | %zu B\n", src_host_ptr,
                     (unsigned)chunk->start_address, (unsigned)offset, size);
  d->copyHostToDevice(src_host_ptr, chunk->start_address + offset, size);
}

void pocl_tce_memfill(void *data, pocl_mem_identifier *dst_mem_id,
                      cl_mem dst_buf, size_t size, size_t offset,
                      const void *__restrict__ pattern, size_t pattern_size) {
  void *__restrict__ device_ptr = dst_mem_id->mem_ptr;
  TCEDevice *d = (TCEDevice *)data;
  chunk_info_t *chunk = (chunk_info_t *)device_ptr;

  void *tmp_memfill_buf = pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, size);
  assert(tmp_memfill_buf);
  pocl_fill_aligned_buf_with_pattern(tmp_memfill_buf, 0, size, pattern,
                                     pattern_size);

  POCL_MSG_PRINT_TCE("MEMFILL %u + %u | %zu B\n",
                     (unsigned)chunk->start_address, (unsigned)offset, size);
  d->copyHostToDevice(tmp_memfill_buf, chunk->start_address + offset, size);

  POCL_MEM_FREE(tmp_memfill_buf);
}

void pocl_tce_read(void *data, void *__restrict__ dst_host_ptr,
                   pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                   size_t offset, size_t size) {
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;
  TCEDevice* d = (TCEDevice*)data;
  chunk_info_t *chunk = (chunk_info_t*)device_ptr;
  POCL_MSG_PRINT_TCE("READ %p <- %u + %u | %zu B\n", dst_host_ptr,
                     (unsigned)chunk->start_address, (unsigned)offset, size);
  d->copyDeviceToHost(chunk->start_address + offset, dst_host_ptr, size);
}

void pocl_tce_copy(void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                   pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                   size_t dst_offset, size_t src_offset, size_t size) {
  void *__restrict__ dst_device_ptr = dst_mem_id->mem_ptr;
  void *__restrict__ src_device_ptr = src_mem_id->mem_ptr;
  TCEDevice *d = (TCEDevice *)data;
  chunk_info_t *src_chunk = (chunk_info_t *)src_device_ptr;
  chunk_info_t *dst_chunk = (chunk_info_t *)dst_device_ptr;
  POCL_MSG_PRINT_TCE("COPY %u + %u -> %u + %u | %zu B\n",
                     (unsigned)src_chunk->start_address, (unsigned)src_offset,
                     (unsigned)dst_chunk->start_address, (unsigned)dst_offset,
                     size);
  d->copyDeviceToDevice(src_chunk->start_address + src_offset,
                        dst_chunk->start_address + dst_offset, size);
}

chunk_info_t*
pocl_tce_malloc_local (void *device_data, size_t size) 
{
  TCEDevice *d = (TCEDevice*)device_data;
  return pocl_alloc_buffer_from_region(&d->local_mem, size);
}

static void pocl_tce_write_kernel_descriptor(_cl_command_node *Command,
                                             cl_kernel kernel,
                                             cl_device_id device,
                                             int Specialize) {
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
          << "void _pocl_kernel_" << meta->name
          << "_workgroup(uint8_t* args, uint8_t* ctx, "
          << "uint32_t, uint32_t, uint32_t);" << std::endl

          << "void _pocl_kernel_" << meta->name
          << "_workgroup_fast(uint8_t* args, uint8_t* ctx, "
          << "uint32_t, uint32_t, uint32_t);" << std::endl

          << "void " << meta->name << "_workgroup_argbuffer("
          << "uint8_t "
          << "__attribute__((address_space(" << device->global_as_id << ")))"
          << "* args, "
          << "uint8_t "
          << "__attribute__((address_space(" << device->global_as_id << ")))"
          << "* ctx, "
          << "uint32_t, uint32_t, uint32_t);" << std::endl;

  if (device->global_as_id != 0)
    content << "__attribute__((address_space(" << device->global_as_id << ")))"
            << std::endl;

  content << "__kernel_metadata _" << meta->name << "_md = {" << std::endl
          << "     \"" << meta->name << "\"," << std::endl
          << "     " << meta->num_args << "," << std::endl
          << "     " << meta->num_locals << "," << std::endl
          << "     " << meta->name << "_workgroup_argbuffer" << std::endl
          << " };" << std::endl;

  pocl_cache_write_descriptor(Command, kernel, Specialize,
                              content.str().c_str(), content.str().size());
}

int pocl_tce_compile_kernel(_cl_command_node *Command, cl_kernel Kernel,
                            cl_device_id Device, int Specialize) {
  if (Command->type != CL_COMMAND_NDRANGE_KERNEL)
    return CL_INVALID_OPERATION;
  _cl_command_run *RunCommand = &Command->command.run;

  void *Data = Command->device->data;
  TCEDevice *Dev = (TCEDevice *)Data;

  if (!Kernel)
    Kernel = Command->command.run.kernel;
  if (!Device)
    Device = Command->device;

  char *Save;
  pocl_sanitize_builtin_kernel_name(Kernel, &Save);

  POCL_LOCK(Dev->tce_compile_lock);
  int Error = pocl_llvm_generate_workgroup_function(
      Command->program_device_i, Device, Kernel, Command, Specialize);

  if (Error) {
    POCL_UNLOCK(Dev->tce_compile_lock);
    POCL_MSG_PRINT_GENERAL("TCE: pocl_llvm_generate_workgroup_function()"
                           " failed for kernel %s\n",
                           Kernel->name);
    return CL_COMPILE_PROGRAM_FAILURE;
  }

  // 12 == strlen (POCL_PARALLEL_BC_FILENAME)
  char ByteCode[POCL_MAX_PATHNAME_LENGTH + 13];

  assert(Dev != NULL);
  assert(Command->command.run.kernel);

  char CacheDir[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_kernel_cachedir_path(CacheDir, Kernel->program,
                                  Command->program_device_i,
                                  Kernel, "", Command, Specialize);
  RunCommand->device_data = strdup(CacheDir);
  POCL_MSG_PRINT_TCE("CACHE DIR: %s\n", CacheDir);

  if (Dev->isNewKernel(RunCommand)) {

    pocl_tce_write_kernel_descriptor(Command, Kernel, Device, Specialize);

    std::string AssemblyFileName(CacheDir);
    TCEString TempDir(CacheDir);
    AssemblyFileName += "/parallel.tpef";

    if (access(AssemblyFileName.c_str(), F_OK) != 0) {
      Error = snprintf(ByteCode, POCL_MAX_PATHNAME_LENGTH + 13, "%s%s",
                       CacheDir, POCL_PARALLEL_BC_FILENAME);
      TCEString BuildCmd = Dev->tceccCommandLine(RunCommand, TempDir, ByteCode,
                                                 AssemblyFileName);

#ifdef DEBUG_TTA_DRIVER
      std::cerr << "CMD: " << BuildCmd << std::endl;
#endif
      POCL_MEASURE_START(TCE_COMPILATION);
      Error = system(BuildCmd.c_str());
      POCL_MEASURE_FINISH(TCE_COMPILATION);
      if (Error != 0) {
        POCL_UNLOCK(Dev->tce_compile_lock);
        POCL_MSG_ERR("Error while running tcecc.\n");
        return CL_COMPILE_PROGRAM_FAILURE;
      }
    }
  }

  pocl_restore_builtin_kernel_name(Kernel, Save);

  POCL_UNLOCK(Dev->tce_compile_lock);
  return CL_SUCCESS;
}

#define CHECK_AND_ALIGN_ARGBUFFER(DSIZE)                                       \
  do {                                                                         \
    if (write_pos + (DSIZE) > last_pos)                                        \
      POCL_ABORT("tce: too many kernel arguments!\n");                         \
    int AlignTarget = MAX_EXTENDED_ALIGNMENT;                                  \
    unsigned T = (intptr_t)write_pos % AlignTarget;                            \
    if (T > 0)                                                                 \
      write_pos += (AlignTarget - T);                                          \
  } while (0)

void
pocl_tce_run(void *data, _cl_command_node* cmd)
{
  assert(cmd->type == CL_COMMAND_NDRANGE_KERNEL);

  TCEDevice *d = (TCEDevice*)data;
  uint32_t kernelAddr;
  unsigned i;
  uint32_t s;

  assert(d != NULL);
  assert(cmd->command.run.kernel);
  assert(cmd->command.run.device_data);

  struct pocl_context *pc = &cmd->command.run.pc;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

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
  dev_cmd.kernel_meta = pocl_byteswap_uint32_t(kernelAddr, d->needsByteSwap);

  const size_t KernArgsSize = 8 * 1024;
  char *temp =
      (char *)pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, KernArgsSize);
  char *write_pos = temp;
  char *last_pos = temp + KernArgsSize;

  struct pocl_argument *al;

  typedef std::vector<chunk_info_t*> ChunkVector;
  /* Chunks to be freed after the kernel finishes. */
  ChunkVector tempChunks;

  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;

  /* these are for the generated standalone code.
   *
   * With argbuffer, all kernel arguments (pointers & scalars) are stored
   * in a single buffer; pointers could be anywhere in that buffer, therefore
   * it's easier to save their position-in-the-argbuffer into temporary array,
   * than to recalculate everything again when we need those positions.*/
  uint32_t gmem_ptr_positions[1024];
  uint32_t gmem_count = 0;

  for (i = 0; i < meta->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          chunk_info_t* local_chunk = pocl_tce_malloc_local (d, al->size);
          if (local_chunk == NULL)
            POCL_ABORT ("Could not allocate memory for a local argument. Out of local mem?\n");

          uint32_t address = pocl_byteswap_uint32_t(local_chunk->start_address,
                                                    d->needsByteSwap);
          POCL_MSG_PRINT_TCE("LOCAL ARG: %u WRITE POS: %p \n", address,
                             write_pos);
          CHECK_AND_ALIGN_ARGBUFFER(4);
          *(uint32_t *)write_pos = address;
          write_pos += 4;

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
          uint32_t address;
          if (al->value == NULL)
            address = 0;
          else {
            assert(al->is_raw_ptr == 0);
            cl_mem m = (*(cl_mem *)(al->value));
            chunk_info_t *p =
                (chunk_info_t *)m->device_ptrs[d->parent->global_mem_id]
                    .mem_ptr;
            address =
                pocl_byteswap_uint32_t(p->start_address, d->needsByteSwap);
          }
          POCL_MSG_PRINT_TCE("PTR ARG: %u WRITE POS: %p\n", address, write_pos);
          CHECK_AND_ALIGN_ARGBUFFER(4);
          if (address)
            gmem_ptr_positions[gmem_count++] = (uint32_t)(write_pos - temp);
          *(uint32_t *)write_pos = address;
          write_pos += 4;
        }
      else /* The scalar values should be byteswapped by the user. */
        {
#ifdef DEBUG_TTA_DRIVER
          printf ("host: copied value from %p to global argument memory\n", al->value);
#endif
          size_t alignment = pocl_size_ceil2(al->size);
          POCL_MSG_PRINT_TCE("SCALAR ARG ALIGN: %u SIZE: %u\n",
                             (unsigned)alignment, (unsigned)al->size);
          CHECK_AND_ALIGN_ARGBUFFER(alignment);
          memcpy(write_pos, al->value, al->size);
          write_pos += al->size;
        }
    }

  /* Allocate the automatic local buffers. */
  for (i = 0; i < meta->num_locals; ++i)
    {
      size_t s = meta->local_sizes[i];
      chunk_info_t* local_chunk = pocl_tce_malloc_local (d, s);
      if (local_chunk == NULL)
        POCL_ABORT ("Could not allocate memory for an automatic local argument. Out of local mem?\n");

      uint32_t address =
          pocl_byteswap_uint32_t(local_chunk->start_address, d->needsByteSwap);
      POCL_MSG_PRINT_TCE("AUTO LOCAL: %u WRITE POS: %p\n", address, write_pos);
      CHECK_AND_ALIGN_ARGBUFFER(4);
      *(uint32_t *)write_pos = address;
      write_pos += 4;

#ifdef DEBUG_TTA_DRIVER
      printf ("host: allocated %zu bytes of local memory for automated local arg %u @ %lu\n",
              s, (meta->num_args + i), local_chunk->start_address);
#endif
      tempChunks.push_back(local_chunk);
    }

    /* Allocate globalmem for kernel args here. */
    s = (write_pos - temp);
    chunk_info_t *kernargs =
        pocl_alloc_buffer_from_region(&d->global_mem, s + 8);
    assert(kernargs);
    POCL_MSG_PRINT_TCE("COPYING %u bytes to KERNARGS: %u \n", s,
                       (uint32_t)kernargs->start_address);
    d->copyHostToDevice(temp, kernargs->start_address, s);

    chunk_info_t *context = pocl_alloc_buffer_from_region(
        &d->global_mem, sizeof(struct pocl_context32));
    pocl_context32 temp_ctx;
    temp_ctx.work_dim =
        pocl_byteswap_uint32_t(cmd->command.run.pc.work_dim, d->needsByteSwap);
    temp_ctx.num_groups[0] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.num_groups[0], d->needsByteSwap);
    temp_ctx.num_groups[1] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.num_groups[1], d->needsByteSwap);
    temp_ctx.num_groups[2] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.num_groups[2], d->needsByteSwap);
    temp_ctx.global_offset[0] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.global_offset[0], d->needsByteSwap);
    temp_ctx.global_offset[1] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.global_offset[1], d->needsByteSwap);
    temp_ctx.global_offset[2] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.global_offset[2], d->needsByteSwap);
    temp_ctx.local_size[0] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.local_size[0], d->needsByteSwap);
    temp_ctx.local_size[1] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.local_size[1], d->needsByteSwap);
    temp_ctx.local_size[2] = pocl_byteswap_uint32_t(
        cmd->command.run.pc.local_size[2], d->needsByteSwap);

    temp_ctx.printf_buffer = pocl_byteswap_uint32_t(
        d->printf_buffer->start_address, d->needsByteSwap);
    temp_ctx.printf_buffer_capacity = pocl_byteswap_uint32_t(
        cmd->device->printf_buffer_size, d->needsByteSwap);
    temp_ctx.printf_buffer_position = pocl_byteswap_uint32_t(
        d->printf_position_chunk->start_address, d->needsByteSwap);
    d->writeWordToDevice(d->printf_position_chunk->start_address, 0);
    POCL_MSG_PRINT_TCE(
        "Device side printf buffer=%x, position: %u and capacity %u\n",
        (unsigned)d->printf_buffer->start_address,
        (int)d->printf_position_chunk->start_address,
        (unsigned)cmd->device->printf_buffer_size);

    POCL_MSG_PRINT_TCE("COPYING %u bytes to CONTEXT: %u \n",
                       (uint32_t)sizeof(struct pocl_context32),
                       (uint32_t)context->start_address);
    d->copyHostToDevice(&temp_ctx, context->start_address,
                        sizeof(struct pocl_context32));

    dev_cmd.status = pocl_byteswap_uint32_t(POCL_KST_FREE, d->needsByteSwap);
    dev_cmd.args =
        pocl_byteswap_uint32_t(kernargs->start_address, d->needsByteSwap);
    dev_cmd.ctx =
        pocl_byteswap_uint32_t(context->start_address, d->needsByteSwap);
    s = sizeof(struct pocl_context32);
    dev_cmd.ctx_size = pocl_byteswap_uint32_t(s, d->needsByteSwap);
    s = write_pos - temp;
    dev_cmd.args_size = pocl_byteswap_uint32_t(s, d->needsByteSwap);
    dev_cmd.kernel_meta = pocl_byteswap_uint32_t(kernelAddr, d->needsByteSwap);
    POCL_MSG_PRINT_TCE("KERNEL %s IS AT: %u \n", kernel->name,
                       dev_cmd.kernel_meta);
    POCL_MSG_PRINT_TCE("ARGS %u   CTX %u   ARG_S %u    CTX_S %u \n",
                       dev_cmd.args, dev_cmd.ctx, dev_cmd.args_size,
                       dev_cmd.ctx_size);

#ifdef DEBUG_TTA_DRIVER
    printf("host: waiting for the device command queue (@ %x) to get room.\n",
           d->statusAddr);
    printf("host: command queue status: %d\n",
           d->readWordFromDevice(d->statusAddr));
#endif
  /* Wait until the device command queue has room. */
    do {
    } while (d->readWordFromDevice(d->statusAddr) != POCL_KST_FREE);

#ifdef DEBUG_TTA_DRIVER
  printf( "host: writing the command.\n");
#endif
  d->copyHostToDevice (&dev_cmd, d->commandQueueAddr, sizeof(__kernel_exec_cmd) );

  /* Ensure the READY status is written the last so the device doesn't
     start executing before all the cmd data has been written. We 
     need a flush or similar mechanism to ensure all the data has 
     been really written, in case the data transfers are not guaranteed
     to be ordered. */
  d->writeWordToDevice(d->statusAddr, POCL_KST_READY);
  dev_cmd.status = pocl_byteswap_uint32_t(POCL_KST_READY, d->needsByteSwap);

  d->notifyKernelRunCommandSent(dev_cmd, &cmd->command.run, gmem_ptr_positions,
                                gmem_count);

#ifdef DEBUG_TTA_DRIVER
  printf("host: commmand queue status: %x\n",
         d->readWordFromDevice(d->statusAddr));

  printf("host: waiting for the command to get executed.\n");
#endif
  /* Wait until the command has executed. */
  unsigned long ticks = 0;
  do {
#ifdef DEBUG_TTA_DRIVER

    if ((ticks % 50) == 0)
      printf("host: commmand queue status: %x\n",
             d->readWordFromDevice(d->statusAddr));
#endif
      usleep(20000);
      ++ticks;
  } while (d->readWordFromDevice(d->statusAddr) != POCL_KST_FINISHED);

#ifdef DEBUG_TTA_DRIVER
  printf( "host: done. Freeing the command queue entry.\n");
#endif
  /* We are done with this kernel, free the command queue entry. */
  d->writeWordToDevice(d->statusAddr, POCL_KST_FREE);

  unsigned printf_position =
      d->readWordFromDevice(d->printf_position_chunk->start_address);
  POCL_MSG_PRINT_TCE(
      "Device wrote %u bytes to printf, passing them to stdout now:\n",
      printf_position);
  if (printf_position > 0) {
    char *tmp_printf_buf = new char[printf_position];
    assert(tmp_printf_buf);
    d->copyDeviceToHost(d->printf_buffer->start_address, tmp_printf_buf,
                        printf_position);
    pocl_write_printf_buffer(tmp_printf_buf, printf_position);
    delete[] tmp_printf_buf;
  }

  for (ChunkVector::iterator i = tempChunks.begin();
       i != tempChunks.end(); ++i) 
    pocl_free_chunk(*i);

  free(temp);
  pocl_free_chunk(kernargs);
  pocl_free_chunk(context);

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
  /* Synch the device global region to the host memory. */
  if (map->map_flags != CL_MAP_WRITE_INVALIDATE_REGION) {
    pocl_tce_read(data, map->host_ptr, src_mem_id, src_buf, map->offset,
                  map->size);
  }

  return CL_SUCCESS;
}

cl_int
pocl_tce_unmap_mem (void *data,
                    pocl_mem_identifier *dst_mem_id,
                    cl_mem dst_buf,
                    mem_mapping_t *map)
{
  if (map->map_flags != CL_MAP_READ) {
    /* Synch the device global region to the host memory. */
    pocl_tce_write (data, map->host_ptr, dst_mem_id, dst_buf, map->offset, map->size);
  }

  return CL_SUCCESS;
}


char* 
pocl_tce_init_build(void *data)
{
  TCEDevice *tce_dev = (TCEDevice*)data;
  TCEString mach_tmpdir =
      Environment::llvmtceCachePath();

  // $HOME/.tce/tcecc/cache may not exist yet, create it here.
  pocl_mkdir_p(mach_tmpdir.c_str());

  TCEString mach_header_base =
      mach_tmpdir + "/" + tce_dev->machine_->hash();

  int error = 0;

  std::string devextHeaderFn =
    std::string(mach_header_base) + std::string("_opencl_devext.h");

  /* Generate the vendor extensions header to provide explicit
     access to the (custom) hardware operations. */
  // to avoid threading issues, generate to tempfile then rename
  if (!pocl_exists(devextHeaderFn.c_str())) {
    char tempfile[POCL_MAX_PATHNAME_LENGTH];
    pocl_mk_tempname(tempfile, mach_tmpdir.c_str(), ".devext", NULL);

    std::string tceopgenCmd = std::string("tceopgen > ") + tempfile;

    POCL_MSG_PRINT_TCE("Running: %s \n", tceopgenCmd.c_str());

    error = system(tceopgenCmd.c_str());
    if (error == -1)
      return NULL;

    std::string extgenCmd = std::string("tceoclextgen ") +
                            tce_dev->machine_file + std::string(" >> ") +
                            tempfile;

    POCL_MSG_PRINT_TCE("Running: %s \n", extgenCmd.c_str());

    error = system(extgenCmd.c_str());
    if (error == -1)
      return NULL;

    // gnu-keywords needed to support the inline asm blocks
    // -fasm doesn't work in the frontend

    pocl_rename(tempfile, devextHeaderFn.c_str());
  }

  std::string includeSwitch =
      std::string("-fgnu-keywords -Dasm=__asm__ -include ") + devextHeaderFn;

  char *include_switch = strdup(includeSwitch.c_str());

  return include_switch;
}

char *
pocl_tce_build_hash (cl_device_id device)
{
  TCEDevice *tce_dev = (TCEDevice*)device->data;
  return strdup(tce_dev->build_hash.c_str());
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

  size_t src_offset = src_origin[0] + src_row_pitch * src_origin[1] + src_slice_pitch * src_origin[2];
  size_t dst_offset = dst_origin[0] + dst_row_pitch * dst_origin[1] + dst_slice_pitch * dst_origin[2];

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
  TCEDevice *d = (TCEDevice *)data;
  chunk_info_t *dst_chunk = (chunk_info_t *)dst_mem_id->mem_ptr;
  size_t adjusted_dst_ptr = dst_chunk->start_address + buffer_origin[0] +
                            buffer_row_pitch * buffer_origin[1] +
                            buffer_slice_pitch * buffer_origin[2];

  char const *__restrict__ const adjusted_host_ptr =
      (char const *)src_host_ptr + host_origin[0] +
      host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  size_t j, k;

  /* TODO: handle overlaping regions */
    
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      {
      size_t s_offset = host_row_pitch * j + host_slice_pitch * k;

      size_t d_offset = buffer_row_pitch * j + buffer_slice_pitch * k;

      d->copyHostToDevice(adjusted_host_ptr + s_offset,
                          adjusted_dst_ptr + d_offset, region[0]);
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
  TCEDevice *d = (TCEDevice *)data;
  chunk_info_t *src_chunk = (chunk_info_t *)src_mem_id->mem_ptr;
  size_t adjusted_src_ptr = src_chunk->start_address + buffer_origin[0] +
                            buffer_row_pitch * buffer_origin[1] +
                            buffer_slice_pitch * buffer_origin[2];

  char const *__restrict__ const adjusted_host_ptr =
      (char const *)dst_host_ptr + host_origin[0] +
      host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  size_t j, k;

  /* TODO: handle overlaping regions */

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      {
      size_t d_offset = host_row_pitch * j + host_slice_pitch * k;
      size_t s_offset = buffer_row_pitch * j + buffer_slice_pitch * k;

      d->copyDeviceToHost(adjusted_src_ptr + s_offset,
                          adjusted_host_ptr + d_offset, region[0]);
      }
}

/*****************************************************************************/
/*****************************************************************************/

int pocl_tce_init_queue(cl_device_id device, cl_command_queue queue) {
  assert(queue->data == NULL);

  tce_queue_data_t *dd = (tce_queue_data_t *)pocl_aligned_malloc(
      HOST_CPU_CACHELINE_SIZE, sizeof(tce_queue_data_t));

  if (dd == NULL)
    goto ERROR;
  queue->data = dd;

  POCL_INIT_COND(dd->cq_cond);

  return CL_SUCCESS;

ERROR:
  pocl_aligned_free(queue->data);
  queue->data = NULL;
  return CL_FAILED;
}

int pocl_tce_free_queue(cl_device_id device, cl_command_queue queue) {
  tce_queue_data_t *qd = (tce_queue_data_t *)queue->data;

  if (queue->data == NULL)
    return CL_SUCCESS;

  POCL_DESTROY_COND(qd->cq_cond);

  POCL_MEM_FREE(queue->data);
  return CL_SUCCESS;
}

/*****************************************************************************/
/*****************************************************************************/

static void tce_push_command(_cl_command_node *node) {
  cl_device_id device = node->device;
  TCEDevice *d = (TCEDevice*)device->data;

  POCL_LOCK(d->wq_lock);
  DL_APPEND(d->work_queue, node);
  POCL_SIGNAL_COND(d->wakeup_cond);
  POCL_UNLOCK(d->wq_lock);
}

void pocl_tce_submit(_cl_command_node *node, cl_command_queue cq) {
  cl_event e = node->sync.event.event;
  assert(e->data == NULL);

  pocl_tce_event_data_t *e_d = NULL;
  e_d = (pocl_tce_event_data_t *)calloc(1, sizeof(pocl_tce_event_data_t));
  assert(e_d);

  POCL_INIT_COND(e_d->event_cond);
  e->data = (void *)e_d;

  node->ready = 1;
  if (pocl_command_is_ready(node->sync.event.event)) {
    pocl_update_event_submitted(node->sync.event.event);
    tce_push_command(node);
  }

  POCL_UNLOCK_OBJ(node->sync.event.event);
  return;
}

void pocl_tce_notify_cmdq_finished(cl_command_queue cq) {
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue */
  tce_queue_data_t *dd = (tce_queue_data_t *)cq->data;
  POCL_BROADCAST_COND(dd->cq_cond);
}

void pocl_tce_join(cl_device_id device, cl_command_queue cq) {
  POCL_LOCK_OBJ(cq);
  tce_queue_data_t *dd = (tce_queue_data_t *)cq->data;

  while (1) {
    if (cq->command_count == 0) {
      POCL_UNLOCK_OBJ(cq);
      return;
    } else {
      POCL_WAIT_COND(dd->cq_cond, cq->pocl_lock);
    }
  }
}

void pocl_tce_flush(cl_device_id device, cl_command_queue cq) {
  // TODO later...
}

void
pocl_tce_notify (cl_device_id device, cl_event event, cl_event finished)
{
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE) {
    pocl_update_event_failed_locked(event);
    return;
  }

  if (!node->ready)
    return;

  if (pocl_command_is_ready(node->sync.event.event)) {
    assert(event->status == CL_QUEUED);
    pocl_update_event_submitted(event);
    tce_push_command(node);
  }

  return;

  POCL_MSG_PRINT_TCE("notify on event %lu \n", event->id);
}

void pocl_tce_wait_event(cl_device_id device, cl_event event) {
  POCL_MSG_PRINT_TCE("device->wait_event on event %lu\n", event->id);
  pocl_tce_event_data_t *e_d = (pocl_tce_event_data_t *)event->data;

  POCL_LOCK_OBJ(event);
  while (event->status > CL_COMPLETE) {
    POCL_WAIT_COND(e_d->event_cond, event->pocl_lock);
  }
  POCL_UNLOCK_OBJ(event);

  POCL_MSG_PRINT_TCE("event wait finished with status: %i\n", event->status);
  assert(event->status <= CL_COMPLETE);
}

void pocl_tce_free_event_data(cl_event event) {
  assert(event->data != NULL);
  pocl_tce_event_data_t *e_d = (pocl_tce_event_data_t *)event->data;
  POCL_DESTROY_COND(e_d->event_cond);
  POCL_MEM_FREE(event->data);
}

void pocl_tce_notify_event_finished(cl_event event) {
  pocl_tce_event_data_t *e_d = (pocl_tce_event_data_t *)event->data;
  POCL_BROADCAST_COND(e_d->event_cond);
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

void *pocl_tce_driver_thread(void *cldev) {
  TCEDevice *d = (TCEDevice *)cldev;

  POCL_LOCK(d->wq_lock);

  while (1) {
    _cl_command_node *cmd;

  RETRY:
    if (d->shutdownRequested) {
      POCL_UNLOCK(d->wq_lock);
      return NULL;
    }

    cmd = d->work_queue;
    if (cmd) {
      DL_DELETE(d->work_queue, cmd);
      POCL_UNLOCK(d->wq_lock);

      assert(pocl_command_is_ready(cmd->sync.event.event));
      assert(cmd->sync.event.event->status == CL_SUBMITTED);

      if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
        pocl_tce_compile_kernel(cmd, cmd->command.run.kernel, cmd->device, 1);

      pocl_exec_command(cmd);

      POCL_LOCK(d->wq_lock);
    }

    if ((d->work_queue == NULL) && (d->shutdownRequested == false)) {
      POCL_WAIT_COND(d->wakeup_cond, d->wq_lock);
      goto RETRY;
    }
  }
}
