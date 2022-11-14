/* XrtDevice.cc - accessing accelerator memory as memory mapped region.

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

#include "XrtDevice.hh"

#include "XrtRegion.hh"
#include "AlmaifShared.hh"

#include "experimental/xrt_kernel.h"

XrtDevice::XrtDevice(char *xrt_kernel_name) {

  unsigned xclbin_char_length = strlen(xrt_kernel_name) + 8;
  char *xclbin_char = (char *)malloc(xclbin_char_length);
  assert(xclbin_char);
  snprintf(xclbin_char, xclbin_char_length, "%s.xclbin", xrt_kernel_name);

  // TODO: Fix the case when the kernel name contains a path
  // Needs to tokenize the last part of the path and use that
  // as the kernel name
  unsigned xrt_kernel_name_length = 2 * strlen(xrt_kernel_name) + 6;
  char *xrt_kernel_name = (char *)malloc(xrt_kernel_name_length);
  assert(xrt_kernel_name);
  snprintf(xrt_kernel_name, xrt_kernel_name_length, "%s:{%s_1}",
           xrt_kernel_name, xrt_kernel_name);

  auto devicehandle = new xrt::device(0);
  assert(devicehandle != NULL && "devicehandle null\n");

  auto uuid = devicehandle->load_xclbin(xclbin_char);
  auto kernel = new xrt::kernel(*devicehandle, uuid, xrt_kernel_name,
                                xrt::kernel::cu_access_mode::exclusive);

  free(xclbin_char);
  free(xrt_kernel_name);

  assert(kernel != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");

  Kernel = (void *)kernel;
  DeviceHandle = (void *)devicehandle;

  ControlMemory = new XrtRegion(0, ALMAIF_DEFAULT_CTRL_SIZE, Kernel);

  discoverDeviceParameters();

  InstructionMemory = new XrtRegion(imem_start, imem_size, Kernel);
  CQMemory = new XrtRegion(cq_start, cq_size, Kernel);
  DataMemory = new XrtRegion(dmem_start, dmem_size, Kernel);
}

XrtDevice::~XrtDevice() {
  delete ((xrt::kernel *)Kernel);
  delete ((xrt::device *)DeviceHandle);
}
