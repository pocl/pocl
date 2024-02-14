/* XilinxXrtDevice.cc - Access AlmaIF device in Xilinx PCIe FPGA.

   Copyright (c) 2022 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "XilinxXrtDevice.hh"

#include "AlmaifShared.hh"
#include "XilinxXrtExternalRegion.hh"
#include "XilinxXrtRegion.hh"

#include "experimental/xrt_ip.h"

#include "pocl_file_util.h"
#include "pocl_timing.h"

#include <libgen.h>

void *DeviceHandle;

XilinxXrtDevice::XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                                 unsigned j) {

  char *TmpKernelName = strdup(XrtKernelNamePrefix.c_str());
  char *KernelName = basename(TmpKernelName);

  std::string xclbin_char = XrtKernelNamePrefix + ".xclbin";

  std::string ExternalMemoryParameters =
      pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "");

  init_xrtdevice(KernelName, xclbin_char, ExternalMemoryParameters, j);

  free(TmpKernelName);
}

XilinxXrtDevice::XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                                 const std::string &XclbinFile, unsigned j) {
  std::string ExternalMemoryParameters =
      pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "");
  init_xrtdevice(XrtKernelNamePrefix, XclbinFile, ExternalMemoryParameters, j);
}

XilinxXrtDevice::XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                                 const std::string &XclbinFile,
                                 const std::string &ExternalMemoryParameters,
                                 unsigned j) {
  init_xrtdevice(XrtKernelNamePrefix, XclbinFile, ExternalMemoryParameters, j);
}

void XilinxXrtDevice::init_xrtdevice(
    const std::string &XrtKernelNamePrefix, const std::string &XclbinFile,
    const std::string &ExternalMemoryParameters, unsigned j) {
  if (j == 0) {
    auto devicehandle = new xrt::device(0);
    assert(devicehandle != NULL && "devicehandle null\n");
    DeviceHandle = (void *)devicehandle;
  }
  programBitstream(XrtKernelNamePrefix, XclbinFile, j);
  // TODO Remove magic
  size_t DeviceOffset = 0x40000000 + j * 0x10000;
  // size_t DeviceOffset = 0x00000000;
  ControlMemory = new XilinxXrtRegion(DeviceOffset, ALMAIF_DEFAULT_CTRL_SIZE,
                                      Kernel, DeviceOffset);

  discoverDeviceParameters();

  char TmpXclbinFile[POCL_MAX_PATHNAME_LENGTH];
  strncpy(TmpXclbinFile, XclbinFile.c_str(), POCL_MAX_PATHNAME_LENGTH);
  char *DirectoryName = dirname(TmpXclbinFile);
  std::string ImgFileName = DirectoryName;
  ImgFileName += "/" + XrtKernelNamePrefix + ".img";
  if (pocl_exists(ImgFileName.c_str())) {
    POCL_MSG_PRINT_ALMAIF(
        "Almaif: Found built-in kernel firmware. Loading it in\n");
    InstructionMemory = new XilinxXrtRegion(ImemStart, ImemSize, Kernel,
                                            ImgFileName, DeviceOffset);
  } else {
    POCL_MSG_PRINT_ALMAIF("Almaif: No default firmware found. Skipping\n");
    InstructionMemory =
        new XilinxXrtRegion(ImemStart, ImemSize, Kernel, DeviceOffset);
  }

  CQMemory = new XilinxXrtRegion(CQStart, CQSize, Kernel, DeviceOffset);
  DataMemory = new XilinxXrtRegion(DmemStart, DmemSize, Kernel, DeviceOffset);

  if (ExternalMemoryParameters != "") {
    char *tmp_params = strdup(ExternalMemoryParameters.c_str());
    char *save_ptr;
    char *param_token = strtok_r(tmp_params, ",", &save_ptr);
    size_t region_address = strtoul(param_token, NULL, 0);
    param_token = strtok_r(NULL, ",", &save_ptr);
    size_t region_size = strtoul(param_token, NULL, 0);
    if (region_size > 0) {
      ExternalXRTMemory = new XilinxXrtExternalRegion(
          region_address, region_size, DeviceHandle);
      POCL_MSG_PRINT_ALMAIF("Almaif: initialized external XRT alloc region at "
                            "%zx with size %zx\n",
                            region_address, region_size);
    }
    free(tmp_params);
  }

  PipeCount_ = 10;
  AllocatedPipes_ = (int *)calloc(PipeCount_, sizeof(int));

  XilinxXrtDeviceInitDone_ = 1;
}

XilinxXrtDevice::~XilinxXrtDevice() {
  delete ((xrt::ip *)Kernel);
  delete ((xrt::device *)DeviceHandle);
  /*  if (ExternalXRTMemory) {
      LL_DELETE(AllocRegions, AllocRegions->next);
    }*/
}

void XilinxXrtDevice::programBitstream(const std::string &XrtKernelNamePrefix,
                                       const std::string &XclbinFile,
                                       unsigned j) {

  xrt::device *devicehandle = (xrt::device *)DeviceHandle;

  // TODO: Fix the case when the kernel name contains a path
  // Needs to tokenize the last part of the path and use that
  // as the kernel name
  std::string XrtKernelName =
      XrtKernelNamePrefix + ":{" + XrtKernelNamePrefix + "_1}";

  if (XilinxXrtDeviceInitDone_) {
    delete (xrt::ip *)Kernel;
  }

  if (j == 0) {
    uint64_t start_time = pocl_gettimemono_ns();
    auto uuid = devicehandle->load_xclbin(XclbinFile);
    uint64_t end_time = pocl_gettimemono_ns();
    printf("Reprogramming done. Time: %d ms\n",
           (end_time - start_time) / 1000000);

    std::string MemInfo = devicehandle->get_info<xrt::info::device::memory>();
    POCL_MSG_PRINT_ALMAIF_MMAP("XRT device's memory info:%s\n",
                               MemInfo.c_str());
  }
  auto uuid = devicehandle->get_xclbin_uuid();

  auto kernel = new xrt::ip(*devicehandle, uuid, XrtKernelName.c_str());

  assert(kernel != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");

  Kernel = (void *)kernel;

  POCL_MSG_PRINT_ALMAIF("TEST\n");
  if (XilinxXrtDeviceInitDone_) {
    ((XilinxXrtRegion *)ControlMemory)->setKernelPtr(Kernel);
    ((XilinxXrtRegion *)InstructionMemory)->setKernelPtr(Kernel);
    ((XilinxXrtRegion *)CQMemory)->setKernelPtr(Kernel);
    ((XilinxXrtRegion *)DataMemory)->setKernelPtr(Kernel);
  }
  POCL_MSG_PRINT_ALMAIF("BITSTREAM PROGRAMMING DONE\n");
}

void XilinxXrtDevice::freeBuffer(pocl_mem_identifier *P) {
  if (P->extra == 1) {
    POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                          ExternalXRTMemory->pointerDeviceOffset(P));
    ExternalXRTMemory->freeBuffer(P);
  } else {
    chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;

    POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                          chunk->start_address);

    assert(chunk != NULL);
    pocl_free_chunk(chunk);
  }
}

size_t XilinxXrtDevice::pointerDeviceOffset(pocl_mem_identifier *P) {
  if (P->extra == 1) {
    return ExternalXRTMemory->pointerDeviceOffset(P);
  } else {
    chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;
    assert(chunk != NULL);
    return chunk->start_address;
  }
}

cl_int XilinxXrtDevice::allocateBuffer(pocl_mem_identifier *P, size_t Size) {

  assert(P->mem_ptr == NULL);
  chunk_info_t *chunk = NULL;

  // TODO: add bufalloc-based on-chip memory allocation here. The current
  // version always allocates from external memory, since the current
  // kernels do not know how to access the on-chip memory.
  if (chunk == NULL) {
    if (ExternalXRTMemory) {
      // XilinxXrtExternalRegion has its own allocation requirements
      // (doesn't use bufalloc)
      cl_int alloc_status = ExternalXRTMemory->allocateBuffer(P, Size);
      P->version = 0;
      P->extra = 1;
      return alloc_status;
    } else {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
  } else {
    POCL_MSG_PRINT_MEMORY("almaif: allocated %zu bytes from 0x%zx\n", Size,
                          chunk->start_address);

    P->mem_ptr = chunk;
    P->extra = 0;
  }
  P->version = 0;
  return CL_SUCCESS;
}

cl_int XilinxXrtDevice::allocatePipe(pocl_mem_identifier *P, size_t Size) {

  assert(P->mem_ptr == NULL);

  P->version = 0;
  int *PipeID = (int *)calloc(1, sizeof(int));
  for (int i = 0; i < PipeCount_; i++) {
    if (AllocatedPipes_[i] == 0) {
      *PipeID = i;
      AllocatedPipes_[i] = 1;
      P->mem_ptr = PipeID;
      POCL_MSG_PRINT_MEMORY("almaif: allocated pipe %i\n", i);
      return CL_SUCCESS;
    }
  }
  return CL_MEM_OBJECT_ALLOCATION_FAILURE;
}

void XilinxXrtDevice::freePipe(pocl_mem_identifier *P) {
  int PipeID = *((int *)P->mem_ptr);
  POCL_MSG_PRINT_MEMORY("almaif: freed pipe %i\n", PipeID);
  AllocatedPipes_[PipeID] = 0;
}

int XilinxXrtDevice::pipeCount() { return PipeCount_; }

void XilinxXrtDevice::writeDataToDevice(pocl_mem_identifier *DstMemId,
                                        const char *__restrict__ const Src,
                                        size_t Size, size_t Offset) {

  if (DstMemId->extra == 0) {
    chunk_info_t *chunk = (chunk_info_t *)DstMemId->mem_ptr;
    size_t Dst = chunk->start_address + Offset;
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to 0x%zx\n", Size, Dst);
    DataMemory->CopyToMMAP(Dst, Src, Size);
  } else if (DstMemId->extra == 1) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to external Xrt buffer\n",
                          Size);
    ExternalXRTMemory->CopyToMMAP(DstMemId, Src, Size, Offset);
  } else {
    POCL_ABORT("Attempt to write data to outside the device memories.\n");
  }
}

void XilinxXrtDevice::readDataFromDevice(char *__restrict__ const Dst,
                                         pocl_mem_identifier *SrcMemId,
                                         size_t Size, size_t Offset) {

  chunk_info_t *chunk = (chunk_info_t *)SrcMemId->mem_ptr;
  POCL_MSG_PRINT_ALMAIF("Reading data with chunk start %zu, and offset %zu\n",
                        chunk->start_address, Offset);
  size_t Src = chunk->start_address + Offset;
  if (SrcMemId->extra == 0) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes from 0x%zx\n", Size, Src);
    DataMemory->CopyFromMMAP(Dst, Src, Size);
  } else if (SrcMemId->extra == 1) {
    POCL_MSG_PRINT_ALMAIF(
        "almaif: Copying %zu bytes from external XRT buffer\n", Size);
    ExternalXRTMemory->CopyFromMMAP(Dst, SrcMemId, Size, Offset);
  } else {
    POCL_ABORT("Attempt to read data from outside the device memories.\n");
  }
}
