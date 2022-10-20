/* MMAPDevice.cc - accessing accelerator memory as memory mapped region.

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

#include "MMAPDevice.hh"

#include "MMAPRegion.hh"
#include "AlmaifShared.hh"

#include "pocl_file_util.h"

#include <unistd.h>
//#include <sys/stat.h>
#include <fcntl.h>

MMAPDevice::MMAPDevice(size_t base_address, char *kernel_name) {
  int mem_fd = -1;
  mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (mem_fd == -1) {
    POCL_ABORT("Could not open /dev/mem\n");
  }
  ControlMemory = new MMAPRegion(base_address, ALMAIF_DEFAULT_CTRL_SIZE, mem_fd);

  discoverDeviceParameters();

  InstructionMemory = new MMAPRegion(imem_start, imem_size, mem_fd);
  CQMemory = new MMAPRegion(cq_start, cq_size, mem_fd);
  DataMemory = new MMAPRegion(dmem_start, dmem_size, mem_fd);

  char file_name[120];
  snprintf(file_name, sizeof(file_name), "%s.img", kernel_name);

  if (pocl_exists(file_name)) {
    POCL_MSG_PRINT_ALMAIF(
        "Almaif: Found built-in kernel firmaware. Loading it in\n");
    ((MMAPRegion *)InstructionMemory)->initRegion(file_name);
  } else {
    POCL_MSG_PRINT_ALMAIF("Almaif: No default firmware found. Skipping\n");
  }

  if (pocl_is_option_set("POCL_ALMAIF_EXTERNALREGION")) {
    char *region_params =
        strdup(pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "0,0"));
    char *save_ptr;
    char *param_token = strtok_r(region_params, ",", &save_ptr);
    size_t region_address = strtoul(param_token, NULL, 0);
    param_token = strtok_r(NULL, ",", &save_ptr);
    size_t region_size = strtoul(param_token, NULL, 0);
    if (region_size > 0) {
      memory_region_t *ext_region =
          (memory_region_t *)calloc(1, sizeof(memory_region_t));
      assert(ext_region && "calloc for ext memory_region_t failed");
      pocl_init_mem_region(ext_region, region_address, region_size);
      LL_APPEND(AllocRegions, ext_region);

      POCL_MSG_PRINT_ALMAIF(
          "Almaif: initialized external alloc region at %zx with size %zx\n",
          region_address, region_size);
      ExternalMemory = new MMAPRegion(region_address, region_size, mem_fd);
    }
    free(region_params);
  }

  close(mem_fd);
}

MMAPDevice::~MMAPDevice() {
  if (ExternalMemory) {
    delete ExternalMemory;
    ExternalMemory = nullptr;
  }
}
