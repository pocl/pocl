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

typedef struct poclcc_kernel_ {
  uint32_t sizeofKernelName;
  char * kernel_name;
  uint32_t sizeofBinary;
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

void poclcc_free(poclcc_global *binary_format);

int poclcc_check_global(poclcc_global *binary_format);
int poclcc_check_device(poclcc_device *device);
uint64_t poclcc_get_device_id(cl_device_id device);
int poclcc_check_device_id(cl_device_id device, poclcc_device *devicecc);
int poclcc_check_binary(cl_device_id device, const unsigned char *binary);

int sizeofPoclccKernel(poclcc_kernel *kernel);
int sizeofPoclccDevice(poclcc_device *device);
int sizeofPoclccGlobal(poclcc_global *binary_format);
int sizeofPoclccGlobalFromBinariesSizes(size_t *binaries_sizes, int num_devices);

cl_int programInfos2BinaryFormat(poclcc_global *binary_format, unsigned char **binaries,
                                 unsigned num_devices);
cl_int binaryFormat2ProgramInfos(unsigned char ***binaries, size_t **binaries_sizes, 
                                 poclcc_global *binary_format);

int binaryFormat2Buffer(char *buffer, int sizeofBuffer, poclcc_global *binary_format);
int buffer2BinaryFormat(poclcc_global *binary_format, char *buffer, int sizeofBuffer);

void poclcc_init_global(poclcc_global *globalcc, int num_devices, poclcc_device *devices);
void poclcc_init_device(poclcc_device *devicecc, cl_device_id device, 
                        int num_kernels, poclcc_kernel *kernels);
void poclcc_init_kernel(poclcc_kernel *kernelcc, char *kernel_name, int sizeofKernelName, 
                        unsigned char *binary, int sizeofBinary);

int LookForKernelBinary(poclcc_global *binary_format, cl_device_id device, char *kernel_name, 
                        char **binary, int *binary_size);

#endif
