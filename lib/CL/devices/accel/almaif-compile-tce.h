
#ifndef POCL_ALMAIFCOMPILETCE_H
#define POCL_ALMAIFCOMPILETCE_H

#include "pocl_util.h"
//#include "accel-shared.h"
//#include "almaif-compile.h"

int pocl_almaif_tce_initialize (cl_device_id device, const char *parameters);
int pocl_almaif_tce_cleanup (cl_device_id device);
int pocl_almaif_tce_compile (_cl_command_node *cmd, cl_kernel kernel,
                             cl_device_id device, int specialize);

typedef struct tce_backend_data_s
{
  pocl_lock_t tce_compile_lock
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  char *machine_file;
  int core_count;
} tce_backend_data_t;

void tceccCommandLine (char *commandline, size_t max_cmdline_len,
                       _cl_command_run *run_cmd, const char *tempDir,
                       const char *inputSrc, const char *outputTpef,
                       const char *machine_file, int is_multicore,
                       int little_endian, const char *extraParams);
void pocl_tce_write_kernel_descriptor (char *content, size_t content_size,
                                       _cl_command_node *command,
                                       cl_kernel kernel, cl_device_id device,
                                       int specialize);

int pocl_almaif_tce_device_hash (const char *adf_file,
                                 const char *llvm_triplet, char *output);

#endif
