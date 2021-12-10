/* MMAPDevice.cc - accessing accelerator memory as memory mapped region.

   Copyright (c) 2021 Pekka Jääskeläinen / Tampere University

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


#include "MMAPDevice.h"

#include "MMAPRegion.h"
#include "accel-shared.h"

#include <unistd.h>
//#include <sys/stat.h>
#include <fcntl.h>




MMAPDevice::MMAPDevice(size_t base_address, char* kernel_name) {
    int mem_fd = -1;  
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd == -1) {
      POCL_ABORT("Could not open /dev/mem\n");
    }
    ControlMemory =
        new MMAPRegion(base_address, ACCEL_DEFAULT_CTRL_SIZE, mem_fd);

    discoverDeviceParameters();

    InstructionMemory = new MMAPRegion(imem_start, imem_size, mem_fd);
    CQMemory = new MMAPRegion(cq_start, cq_size, mem_fd);
    DataMemory = new MMAPRegion(dmem_start, dmem_size, mem_fd);

    char file_name[120];
    snprintf(file_name, sizeof(file_name), "%s.img", kernel_name);


    ((MMAPRegion*)InstructionMemory)->initRegion(file_name); 
    close(mem_fd);
}


