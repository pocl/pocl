/* cellspu.h - a pocl device driver for Cell SPU.

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

#ifndef POCL_CELLSPU_H
#define POCL_CELLSPU_H

#include "pocl_cl.h"
#include "pocl_icd.h"
#include "bufalloc.h"

#include "prototypes.inc"

/* simplistic linker script: 
 * this is the SPU local address where 'OpenCL global' memory starts.
 * (if we merge the spus to a single device, this is the 'OpenCL local' memory
 * 
 * The idea is to allocate
 * 64k (0-64k) for text.
 * 128k (64k-192k) for Opencl local memory.
 * 64k (192k-256k) for stack + heap (if any)
 * 
 * I was unable to place the stack to start at 0x20000, thus the "unclean" division.
 */
#define CELLSPU_OCL_BUFFERS_START 0x10000
#define CELLSPU_OCL_BUFFERS_SIZE  0x20000
#define CELLSPU_KERNEL_CMD_ADDR   0x30000
//#define CELLSPU_OCL_KERNEL_ADDRESS 0x2000


#ifdef __cplusplus
extern "C" {
#endif

GEN_PROTOTYPES (cellspu)
GEN_PROTOTYPES (basic)

#ifdef __cplusplus
}
#endif

#endif /* POCL_CELLSPU_H */
