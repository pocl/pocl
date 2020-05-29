/* pocl-ptx-gen.h - declarations for PTX code generator

   Copyright (c) 2016-2017 James Price / University of Bristol

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

#ifndef POCL_PTX_GEN_H
#define POCL_PTX_GEN_H

#include "config.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Search for the libdevice bitcode library for the given GPU architecture. */
/* Returns zero on success, non-zero on failure. */
int findLibDevice(char LibDevicePath[PATH_MAX], const char *Arch);

/* Generate a PTX file from an LLVM bitcode file. */
/* Returns zero on success, non-zero on failure. */
int pocl_ptx_gen (const char *BitcodeFilename, const char *PTXFilename,
                  const char *KernelName,
                  const char *Arch, const char *LibDevicePath, int HasOffsets);

/* Populate the Alignments array with the required pointer alignments for */
/* each kernel argument. */
/* Returns zero on success, non-zero on failure. */
int pocl_cuda_get_ptr_arg_alignment(const char *BitcodeFilename,
                                    const char *KernelName,
                                    size_t *Alignments);

#ifdef __cplusplus
}
#endif

#endif /* POCL_PTX_GEN_H */
