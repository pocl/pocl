/* tce_common.cc - common functionality over the different TCE/TTA device drivers.

   Copyright (c) 2012-2013 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "config.h"
#include "install-paths.h"

#include <unistd.h>

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

using namespace TTAMachine;

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128
#define max(a,b) (((a) > (b)) ? (a) : (b))

#define ALIGNMENT (max(ALIGNOF_FLOAT16, ALIGNOF_DOUBLE16))

//#define DEBUG_TTA_DRIVER

TCEDevice::TCEDevice(cl_device_id dev, const char* adfName) :
  local_as(NULL), global_as(NULL), private_as(NULL), machine_file(adfName), parent(dev),
  currentProgram(NULL), curKernelAddr(0), curKernel(NULL), globalCycleCount(0) {
  parent->data = this;
#if defined(WORDS_BIGENDIAN) && WORDS_BIGENDIAN == 1
  needsByteSwap = false;
#else
  needsByteSwap = true;
#endif  
}

TCEDevice::~TCEDevice() {
  parent->data = NULL;
}

bool
TCEDevice::isMultiCoreMachine() const {
#if defined(TCEMC_AVAILABLE) && TCEMC_AVAILABLE == 1
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
  
  try {
    commandQueueAddr = globalScope.dataLabel("kernel_command").address().location();
  } catch (const KeyNotFound& e) {
    POCL_ABORT ("Could not find the shared data structures from the device binary.");
  }    
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
    (&global_mem, (memory_address_t)global_as->start() + TTA_UNALLOCATED_GLOBAL_SPACE, 
     parent->global_mem_size);
}

TCEString
TCEDevice::tceccCommandLine
(_cl_command_run *run_cmd, const TCEString& inputSrc, 
 const TCEString& outputTpef, const TCEString extraParams) 
{

  TCEString mainC;
  if (isMultiCoreMachine()) 
    mainC = "tta_device_main_dthread.c";
  else
    mainC = "tta_device_main.c";

  TCEString deviceMainSrc;
  TCEString poclIncludePathSwitch;
  if (getenv("POCL_BUILDING") != NULL)
    {
      deviceMainSrc = TCEString(BUILDDIR) + "/lib/CL/devices/tce/" + mainC;
      poclIncludePathSwitch = " -I " SRCDIR "/include";
    }
  else 
    {
      deviceMainSrc = TCEString(PKGDATADIR) + "/" + mainC;
      assert(access(deviceMainSrc.c_str(), R_OK) == 0);
      poclIncludePathSwitch = " -I " PKGINCLUDEDIR;
    }

  TCEString extraFlags = extraParams;
  if (isMultiCoreMachine())
    extraFlags += " -ldthread -lsync-lu -llockunit";

  TCEString tempDir = run_cmd->tmp_dir;

  std::string kernelObjSrc = "";
  kernelObjSrc += tempDir;
  kernelObjSrc += "/../descriptor.so.kernel_obj.c";

  if (getenv("POCL_TCECC_EXTRA_FLAGS") != NULL)
    extraFlags += " " + TCEString(getenv("POCL_TCECC_EXTRA_FLAGS"));

  std::string userProgramBuildOptions;
  if (run_cmd->kernel->program->compiler_options != NULL)
    userProgramBuildOptions = run_cmd->kernel->program->compiler_options;

  std::string kernelMdSymbolName = "_";
  kernelMdSymbolName += run_cmd->kernel->name;
  kernelMdSymbolName += "_md";

  TCEString programBcFile = tempDir + "/program.bc";
  /* Compile in steps to save the program.bc for automated exploration 
     use case when producing the kernel capture scripts. */
  TCEString cmdLine;
  cmdLine << "tcecc --vector-backend -llwpr " + poclIncludePathSwitch + " " + deviceMainSrc + " " + 
    userProgramBuildOptions + " " + kernelObjSrc + " " + inputSrc + 
    " -k " + kernelMdSymbolName +
    " -g -O3 --emit-llvm -o " + programBcFile + " " + extraFlags + ";";

  cmdLine << "tcecc $* --vector-backend -a " << machine_file << " " << programBcFile 
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
  printf("host: malloc %x (host) %d (device) size: %u\n", host_ptr, chunk->start_address, size);
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

void
pocl_tce_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  TCEDevice* d = (TCEDevice*)data;
  chunk_info_t *chunk = (chunk_info_t*)device_ptr;
#ifdef DEBUG_TTA_DRIVER
  printf("host: write %x %x %u\n", host_ptr, chunk->start_address, cb);
#endif
  d->copyHostToDevice(host_ptr, chunk->start_address, cb);
}

void
pocl_tce_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  TCEDevice* d = (TCEDevice*)data;
  chunk_info_t *chunk = (chunk_info_t*)device_ptr;
#ifdef DEBUG_TTA_DRIVER
  printf("host: read to %x (host) from %d (device) %u\n", host_ptr, chunk->start_address, cb);
#endif
  d->copyDeviceToHost(chunk->start_address, host_ptr, cb);
}

void *
pocl_tce_create_sub_buffer (void */*device_data*/, void* buffer, size_t origin, size_t size)
{
#ifdef DEBUG_TTA_DRIVER
  printf("host: create sub buffer %d (buf start) + %d size: %d\n", 
         ((chunk_info_t*)buffer)->start_address, origin, size);
#endif

  return create_sub_chunk ((chunk_info_t*)buffer, origin, size);
}

chunk_info_t*
pocl_tce_malloc_local (void *device_data, size_t size) 
{
  TCEDevice *d = (TCEDevice*)device_data;
  return alloc_buffer (&d->local_mem, size);
}

void
pocl_tce_free (void */*data*/, cl_mem_flags /*flags*/, void *ptr)
{
  free_chunk ((chunk_info_t*) ptr);
}

void 
pocl_tce_run 
(void *data, 
 _cl_command_node* cmd) {

  TCEDevice *d = (TCEDevice*)data;
  int error;
  char bytecode[POCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];
  uint32_t kernelAddr;
  unsigned i;

  assert (data != NULL);

  if (d->isNewKernel(&(cmd->command.run))) {
    std::string assemblyFileName(cmd->command.run.tmp_dir);
    assemblyFileName += "/parallel.tpef";
    
    std::string kernelMdSymbolName = "_";
    kernelMdSymbolName += cmd->command.run.kernel->name;
    kernelMdSymbolName += "_md";
    
    std::string userProgramBuildOptions;
    if (cmd->command.run.kernel->program->compiler_options != NULL)
      userProgramBuildOptions = cmd->command.run.kernel->program->compiler_options;
    
    if (access (assemblyFileName.c_str(), F_OK) != 0)
      {
        error = snprintf (bytecode, POCL_FILENAME_LENGTH,
                          "%s/%s", cmd->command.run.tmp_dir, POCL_PARALLEL_BC_FILENAME);
        TCEString buildCmd = 
          d->tceccCommandLine(&cmd->command.run, bytecode, assemblyFileName);
        
#ifdef DEBUG_TTA_DRIVER
      std::cerr << "CMD: " << buildCmd << std::endl;
#endif
      error = system(buildCmd.c_str());
      if (error != 0)
        POCL_ABORT("Error while running tcecc.");
      }
    
    try {
      d->loadProgramToDevice(assemblyFileName);
      d->restartProgram();
    } catch (Exception &e) {
      std::cerr << "error: " << e.errorMessage() << std::endl;
      POCL_ABORT("error: Failed to load program to the TTA.");
    }

    const TTAProgram::Program* prog = d->currentProgram;
    assert (prog != NULL);
    
    const TTAProgram::GlobalScope& globalScope = prog->globalScopeConst();
    
    try {
      kernelAddr = globalScope.dataLabel(kernelMdSymbolName).address().location();
    } catch (const KeyNotFound& e) {
      POCL_ABORT ("Could not find the shared data structures from the device binary.");
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

  for (i = 0; i < cmd->command.run.kernel->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (cmd->command.run.kernel->arg_is_local[i])
        {
          chunk_info_t* local_chunk = pocl_tce_malloc_local (d, al->size);
          if (local_chunk == NULL)
            POCL_ABORT ("Could not allocate memory for a local argument. Out of local mem?\n");

          dev_cmd.args[i] = byteswap_uint32_t (local_chunk->start_address, d->needsByteSwap);
#ifdef DEBUG_TTA_DRIVER
          printf ("host: allocated %d bytes of local memory for arg %d @ %d\n", 
                  al->size, i, local_chunk->start_address);
#endif
          tempChunks.push_back(local_chunk);
        }
      else if (cmd->command.run.kernel->arg_is_pointer[i])
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            dev_cmd.args[i] = 0;
          else
            dev_cmd.args[i] = byteswap_uint32_t 
              (((chunk_info_t*)((*(cl_mem *) (al->value))->device_ptrs[d->parent->dev_id]))->start_address, d->needsByteSwap);
        }
      else /* The scalar values should be byteswapped by the user. */
        {
          /* Copy the scalar argument data to the shared memory. */
          chunk_info_t* arg_space = 
            (chunk_info_t*)pocl_tce_malloc (d, CL_MEM_COPY_HOST_PTR, al->size, al->value);
          if (arg_space == NULL)
            POCL_ABORT ("Could not allocate memory from the device argument space. Out of global mem?\n");
#ifdef DEBUG_TTA_DRIVER
          printf ("host: copied value from %x to global argument memory\n", al->value);
#endif
          dev_cmd.args[i] = byteswap_uint32_t (arg_space->start_address, d->needsByteSwap);
          tempChunks.push_back(arg_space);
        }
    }

  /* Allocate the automatic local buffers. */
  for (std::size_t i = cmd->command.run.kernel->num_args;
       i < cmd->command.run.kernel->num_args + cmd->command.run.kernel->num_locals;
       ++i) 
    {
      al = &(cmd->command.run.arguments[i]);
      chunk_info_t* local_chunk = pocl_tce_malloc_local (d, al->size);
      if (local_chunk == NULL)
        POCL_ABORT ("Could not allocate memory for an automatic local argument. Out of local mem?\n");

      dev_cmd.args[i] = byteswap_uint32_t (local_chunk->start_address, d->needsByteSwap);
#ifdef DEBUG_TTA_DRIVER
      printf ("host: allocated %d bytes of local memory for automated local arg %d @ %d\n", 
              al->size, i, local_chunk->start_address);
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
  } while (d->readWordFromDevice(d->commandQueueAddr) != POCL_KST_FINISHED);

#ifdef DEBUG_TTA_DRIVER
  printf( "host: done. Freeing the command queue entry.\n");
#endif
  /* We are done with this kernel, free the command queue entry. */
  d->writeWordToDevice(d->commandQueueAddr, POCL_KST_FREE);

  for (ChunkVector::iterator i = tempChunks.begin(); 
       i != tempChunks.end(); ++i) 
    free_chunk (*i);

#ifdef DEBUG_TTA_DRIVER
  printf("host: local memory allocations:\n");
  print_chunks (d->local_mem.chunks);

  printf("host: global memory allocations:\n");
  print_chunks (d->global_mem.chunks);
#endif  
}

void *
pocl_tce_map_mem (void *data, void *buf_ptr, 
                  size_t /*offset*/, size_t size,
                  void *host_ptr) 
{
  void *target = NULL;
  chunk_info_t *chunk = (chunk_info_t*)buf_ptr;
  if (host_ptr != NULL) 
    {
      target = host_ptr;
    } 
  else
    {
      posix_memalign (&target, ALIGNMENT, size);
    }

  /* Synch the device global region to the host memory. */
  pocl_tce_read (data, target, chunk, size);
  return target;
}

char* 
pocl_tce_init_build(void *data, const char *dev_tmpdir) 
{
  TCEDevice *tce_dev = (TCEDevice*)data;

  int error = 0;
  /* Generate the vendor extensions header to provide explicit
     access to the (custom) hardware operations. */
  std::string tceopgenCmd = 
    std::string("tceopgen > ") + std::string(dev_tmpdir) + "/tceops.h";
  
  error = system (tceopgenCmd.c_str());
  if (error == -1) return NULL;

  std::string devextHeaderFn =
    std::string(dev_tmpdir) + std::string("/_devext.h");

  std::string extgenCmd = 
    std::string("tceoclextgen ") + tce_dev->machine_file + 
    std::string(" > ") + devextHeaderFn;

  error = system (extgenCmd.c_str());
  if (error == -1) return NULL;

  // gnu-keywords needed to support the inline asm blocks
  // -fasm doesn't work in the frontend

  std::string includeSwitch = 
    std::string("-fgnu-keywords -Dasm=__asm__ -include ") + devextHeaderFn;
  
  char *include_switch = strdup(includeSwitch.c_str());

  return include_switch;
}

int 
pocl_tce_build_program (void *data, const char *source_fn, const char *binary_fn, 
                        const char *default_cmd, const char *user_opts, const char *dev_tmpdir) 
{
  const char *include_switch =
    pocl_tce_init_build(data, dev_tmpdir);

  std::string buildCmd = 
    std::string("EXTRA_CPPFLAGS=\"") + std::string(include_switch) +
    std::string("\" ") + std::string(default_cmd);

  free ((void*)include_switch);

  return system (buildCmd.c_str());
}

void
pocl_tce_copy (void */*data*/, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb)
{
  POCL_ABORT_UNIMPLEMENTED();
  if (src_ptr == dst_ptr)
    return;
  
  memcpy (dst_ptr, src_ptr, cb);
}

void
pocl_tce_copy_rect (void */*data*/,
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
pocl_tce_write_rect (void */*data*/,
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
pocl_tce_read_rect (void */*data*/,
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
