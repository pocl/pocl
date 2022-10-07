/* cpuinfo.h - parsing of /proc/cpuinfo for OpenCL device info

   Copyright (c) 2012 Pekka Jääskeläinen
   
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
/**
 * Functionality for using the hwloc library for automatically detecting
 * the device characteristics and filling the info to the device structure to
 * make the info accessible to the clGetDeviceInfo() etc.
 *
 * http://www.open-mpi.org/projects/hwloc/
 */
#ifndef POCL_CPUINFO_H
#define POCL_CPUINFO_H

#include "pocl_cl.h"

POCL_EXPORT
void pocl_cpuinfo_detect_device_info(cl_device_id device);

POCL_EXPORT
void fix_local_mem_size (cl_device_id device);
#endif /* POCL_TOPOLOGY_H */
