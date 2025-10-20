/* pocl_mlir.h: interface to call mlir and Clang.

   Copyright (c) 2025 Topi Leppänen / Tampere University

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

#ifndef POCL_MLIR_H
#define POCL_MLIR_H

#include "pocl_cl.h"

typedef struct PoclLLVMContextData PoclLLVMContextData;

#ifdef __cplusplus
extern "C"
{
#endif
POCL_EXPORT
int poclMlirGenerateStandardWorkgroupFunction (unsigned DeviceI,
                                               cl_device_id Device,
                                               cl_kernel Kernel,
                                               _cl_command_node *Command,
                                               int Specialize,
                                               const char *Cachedir);

/* Compiles an .cl file into LLVM IR.
 */
POCL_EXPORT
int poclMlirBuildProgram (cl_program Program,
                          unsigned DeviceI,
                          cl_uint NumInputHeaders,
                          const cl_program *InputHeaders,
                          const char **HeaderIncludeNames,
                          int LinkingProgram);

POCL_EXPORT
unsigned poclMlirGetKernelCount (cl_program Program, unsigned DeviceI);

/* Retrieve metadata of the given kernel in the program to populate the
 * cl_kernel object.
 */
POCL_EXPORT
int poclMlirGetKernelsMetadata (cl_program Program, unsigned DeviceI);

POCL_EXPORT
int poclMlirGenerateLlvmFunction (unsigned DeviceI,
                                  cl_device_id Device,
                                  cl_kernel Kernel,
                                  _cl_command_node *Command,
                                  int Specialize,
                                  const char *Cachedir);

POCL_EXPORT
void poclDestroyMlirModule (void *Module);

POCL_EXPORT
int poclMlirGenerateWorkgroupFunctionNowrite (unsigned DeviceI,
                                              cl_device_id Device,
                                              cl_kernel Kernel,
                                              _cl_command_node *Command,
                                              void **Output,
                                              int Specialize,
                                              cl_program Program);

void poclMlirRegisterDialects (PoclLLVMContextData *Data);

#ifdef __cplusplus
}
#endif

#endif
