/* AlmaifCompileTCE.hh - compiler support for custom devices

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

#ifndef POCL_ALMAIFCOMPILETCE_H
#define POCL_ALMAIFCOMPILETCE_H

#include "pocl_util.h"
//#include "AlmaifShared.hh"
//#include "AlmaifCompile.hh"

int pocl_almaif_tce_initialize(cl_device_id device, const char *parameters);
int pocl_almaif_tce_cleanup(cl_device_id device);
void pocl_almaif_tce_compile(_cl_command_node *cmd, cl_kernel kernel,
                             cl_device_id device, int specialize);
void pocl_almaif_tce_produce_standalone_program(AlmaifData *D,
                                                _cl_command_node *cmd,
                                                pocl_context32 *pc,
                                                size_t arg_size,
                                                void *arguments);

char *pocl_tce_init_build(void *data);

typedef struct tce_backend_data_s {
  pocl_lock_t tce_compile_lock
      __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  char *machine_file;
  int core_count;
} tce_backend_data_t;

void tceccCommandLine(char *commandline, size_t max_cmdline_len,
                      _cl_command_run *run_cmd, AlmaifData *D,
                      const char *tempDir, const char *inputSrc,
                      const char *outputTpef, const char *machine_file,
                      int is_multicore, int little_endian,
                      const char *extraParams, bool standalone_mode);
void pocl_tce_write_kernel_descriptor(char *content, size_t content_size,
                                      _cl_command_node *command,
                                      cl_kernel kernel, cl_device_id device,
                                      int specialize);

int pocl_almaif_tce_device_hash(const char *adf_file, const char *llvm_triplet,
                                char *output);

void set_preprocessor_directives(char *output, AlmaifData *d, const char *adf,
                                 bool standalone_mode);

#endif
