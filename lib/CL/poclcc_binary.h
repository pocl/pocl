#ifndef POCL_BINARY_FORMAT_H
#define POCL_BINARY_FORMAT_H

#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "pocl_cl.h"

/* Binary format identifier */
#define POCLCC_STRING_ID "poclbin"
#define POCLCC_STRING_ID_LENGTH 7
#define POCLCC_VERSION 1  

/* Binary format internal structures */
typedef struct poclcc_kernel_ {
  uint32_t sizeofKernelName;
  uint32_t num_args;
  uint32_t num_locals;
  uint32_t sizeofBinary;
  char * kernel_name;
  struct pocl_argument *dyn_arguments;
  struct pocl_argument_info *arg_info;
  unsigned char *binary;
} poclcc_kernel;

typedef struct poclcc_device_ {
  char pocl_id[POCLCC_STRING_ID_LENGTH];
  uint64_t device_id;
  uint32_t num_kernels;
  poclcc_kernel *kernels;
} poclcc_device;

typedef struct poclcc_global_ {
  char pocl_id[POCLCC_STRING_ID_LENGTH];
  uint32_t version;
  uint32_t num_devices;
  poclcc_device *devices;
} poclcc_global;

// free internal structures
void poclcc_free(poclcc_global *binary_format);
void poclcc_device_free(poclcc_device *device);
void poclcc_kernel_free(poclcc_kernel *kernel);

// check binary format validity
int poclcc_check_global(poclcc_global *binary_format);
int poclcc_check_device(poclcc_device *device);
uint64_t poclcc_get_device_id(cl_device_id device);
int poclcc_check_device_id(cl_device_id device, poclcc_device *devicecc);
int poclcc_check_binary(cl_device_id device, const unsigned char *binary);

// get size of struct (as a continuous buffer)
int poclcc_sizeofKernel(poclcc_kernel *kernel);
int poclcc_sizeofDevice(poclcc_device *device);
int poclcc_sizeofGlobal(poclcc_global *binary_format);
int poclcc_sizeofGlobalFromBinariesSizes(size_t *binaries_sizes, int num_devices);

// conversion from/to binary format
cl_int poclcc_programInfos2BinaryFormat(poclcc_global *binary_format, unsigned char **binaries,
                                 unsigned num_devices);
cl_int poclcc_binaryFormat2ProgramInfos(unsigned char ***binaries, size_t **binaries_sizes, 
                                 poclcc_global *binary_format);

int poclcc_binaryFormat2Buffer(char *buffer, int sizeofBuffer, poclcc_global *binary_format);
int poclcc_buffer2BinaryFormat(poclcc_global *binary_format, char *buffer, int sizeofBuffer);

// initialize cl_kernel data from a poclcc_kernel specify with kernel_name and cl_device_id
cl_int poclcc_binaryFormat2ClKernel(poclcc_global *binary_format, const char *kernel_name,
                             cl_kernel kernel, cl_device_id device);

// initialize internal struct
void poclcc_init_global(poclcc_global *globalcc, int num_devices, poclcc_device *devices);
void poclcc_init_device(poclcc_device *devicecc, cl_device_id device, 
                        int num_kernels, poclcc_kernel *kernels);
void poclcc_init_kernel(poclcc_kernel *kernelcc, char *kernel_name, int sizeofKernelName, 
                        unsigned char *binary, int sizeofBinary, int num_args, int num_locals,
                        struct pocl_argument *dyn_arguments, struct pocl_argument_info *arg_info);

// look for a binary with kernel_name and cl_device_id
int poclcc_LookForKernelBinary(poclcc_global *binary_format, cl_device_id device, const char *kernel_name, 
                        char **binary, int *binary_size);

#endif
