#include "pocl_binary_format.h"

void poclcc_free(poclcc_global *binary_format){
  int i;
  if (binary_format->devices != NULL){
    for (i=0; i<binary_format->num_devices; i++){
      poclcc_device *device = &(binary_format->devices[i]);
      if (device != NULL){
        if (device->kernels != NULL){
          int j;
          for (j=0; j<device->num_kernels; j++){
            poclcc_kernel *kernel = &(device->kernels[j]);
            if (kernel != NULL){
              POCL_MEM_FREE(kernel->kernel_name);
              POCL_MEM_FREE(kernel->binary);
            }
          }
          POCL_MEM_FREE(device->kernels);
        }
      }
    }
    POCL_MEM_FREE(binary_format->devices);
  }
}

int poclcc_check_global(poclcc_global *binary_format){
  if (binary_format->version != POCLCC_VERSION)
    return 0;
  return !strncmp(binary_format->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
}

int poclcc_check_device(poclcc_device *device){
  return !strncmp(device->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
}

#define FNV_OFFSET 0xcbf29ce484222325
#define FNV_PRIME 0x100000001b3
uint64_t poclcc_get_device_id(cl_device_id device){
  //FNV-1A with vendor_id, llvm_target_triplet and llvm_cpu
  uint64_t result = FNV_OFFSET;
  const char *llvm_tt = device->llvm_target_triplet;
  const char *llvm_cpu = device->llvm_cpu;

  result *= 0x100000001b3;
  result ^= device->vendor_id;
  int i, length = strlen(llvm_tt);
  for (i=0; i<length; i++){
    result *= FNV_PRIME;
    result ^= llvm_tt[i];
  }
  length = strlen(llvm_cpu);
  for (i=0; i<length; i++){
    result *= FNV_PRIME;
    result ^= llvm_cpu[i];
  }

  return result;
}

int poclcc_check_device_id(cl_device_id device, poclcc_device *devicecc){
  return poclcc_get_device_id(device) == devicecc->device_id;
}

int poclcc_check_binary(cl_device_id device, const unsigned char *binary){
  poclcc_device *devicecc = (poclcc_device *)binary;
  return poclcc_check_device(devicecc) && poclcc_check_device_id(device, devicecc);
}




int sizeofPoclccKernel(poclcc_kernel *kernel){
  return sizeof(poclcc_kernel) + kernel->sizeofKernelName + kernel->sizeofBinary
    -sizeof(kernel->kernel_name) - sizeof(kernel->binary);
}

int sizeofPoclccDevice(poclcc_device *device){
  int size = sizeof(poclcc_device) - sizeof(device->kernels);
  int i;
  for (i=0; i<device->num_kernels; i++){
    size += sizeofPoclccKernel(&(device->kernels[i]));
  }
  return size;
}

int sizeofPoclccGlobal(poclcc_global *binary_format){
  int size = 0;
  if (binary_format == NULL)
    return -1;
    
  size += sizeof(poclcc_global) - sizeof(poclcc_device *);
  int i;
  for (i=0; i<binary_format->num_devices; i++)
    size += sizeofPoclccDevice(&(binary_format->devices[i]));
  
  return size;
}

int sizeofPoclccGlobalFromBinariesSizes(size_t *binaries_sizes, int num_devices){
  if (binaries_sizes == NULL)
    return -1;

  int i;
  int size = 0;
  for (i=0; i<num_devices; i++){
    size += binaries_sizes[i];
  }
  size += sizeof(poclcc_global) - sizeof(poclcc_device *);
  return size;
}




cl_int programInfos2BinaryFormat(poclcc_global *binary_format, unsigned char **binaries,
                                 unsigned num_devices){
  strncpy(binary_format->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  binary_format->version = POCLCC_VERSION;
  binary_format->num_devices = num_devices;
  if ((binary_format->devices = malloc(num_devices*sizeof(poclcc_device))) == NULL){
    return CL_OUT_OF_HOST_MEMORY;
  }

  int i;
  for (i=0; i<num_devices; i++){
    poclcc_device *device_dst = &(binary_format->devices[i]);
    poclcc_device *device_src = (poclcc_device *)(binaries[i]);
    memcpy(device_dst, device_src, sizeof(poclcc_device));    
    assert(poclcc_check_device(device_dst));
    
    int num_kernels = device_dst->num_kernels;
    if ((device_dst->kernels = malloc(num_kernels*sizeof(poclcc_kernel))) == NULL){
      poclcc_free(binary_format);
      return CL_OUT_OF_HOST_MEMORY;
    }
    int j;
    char *kernel_src = (char *)(&(device_src->kernels));
    for (j=0; j<num_kernels; j++){
      poclcc_kernel *kernel_dst = &(device_dst->kernels[i]);

      int sizeofKernelName = *((int *)kernel_src);
      kernel_dst->sizeofKernelName = sizeofKernelName;
      kernel_src += sizeof(int);

      if ((kernel_dst->kernel_name = malloc(sizeofKernelName)) == NULL){
        poclcc_free(binary_format);
        return CL_OUT_OF_HOST_MEMORY;
      }
      memcpy(kernel_dst->kernel_name, kernel_src, sizeofKernelName);
      kernel_src += sizeofKernelName;

      int sizeofBinary = *((int *)kernel_src);
      kernel_dst->sizeofBinary = sizeofBinary;
      kernel_src += sizeof(int);

      if ((kernel_dst->binary = malloc(sizeofBinary)) == NULL){
        poclcc_free(binary_format);
        return CL_OUT_OF_HOST_MEMORY;
      }
      memcpy(kernel_dst->binary, kernel_src, sizeofBinary);
      kernel_src += sizeofBinary;
    }
  }
  return CL_SUCCESS;
}

cl_int binaryFormat2ProgramInfos(unsigned char ***binaries, size_t **binaries_sizes, 
                                 poclcc_global *binary_format){
  assert(poclcc_check_global(binary_format));
  int num_devices = binary_format->num_devices;

  if ((*binaries_sizes = malloc(sizeof(size_t)*num_devices)) == NULL)
    goto ERROR;
  
  if ((*binaries = malloc(sizeof(unsigned char*)*num_devices)) == NULL)
    goto ERROR_CLEAN_BINARIES_SIZES;
  
  int i;
  for (i=0; i<num_devices; i++){
    unsigned char *devicecc_binary;
    poclcc_device *device = &(binary_format->devices[i]);
    int sizeofDevicecc = sizeofPoclccDevice(device);
    if ((devicecc_binary = malloc(sizeofDevicecc)) == NULL)
      goto ERROR_CLEAN_BINARIES_ALL;
    (*binaries)[i] = devicecc_binary;
    (*binaries_sizes)[i] = sizeofDevicecc;

    memcpy(devicecc_binary, device, sizeof(poclcc_device));
    devicecc_binary += sizeof(poclcc_device) - sizeof(poclcc_kernel *);
    
    int j;
    for (j=0; j<device->num_kernels; j++){
      poclcc_kernel *kernel = &(device->kernels[j]);

      int sizeofKernelName = kernel->sizeofKernelName;
      *((int *)devicecc_binary) = sizeofKernelName;
      devicecc_binary += sizeof(int);

      memcpy(devicecc_binary, kernel->kernel_name, sizeofKernelName);
      devicecc_binary += sizeofKernelName;

      int sizeofBinary = kernel->sizeofBinary;
      *((int *)devicecc_binary) = sizeofBinary;
      devicecc_binary += sizeof(int);

      memcpy(devicecc_binary, kernel->binary, sizeofBinary);
      devicecc_binary += sizeofBinary;
    }
  }
  return CL_SUCCESS;

ERROR_CLEAN_BINARIES_ALL:
  for (i=0; i<num_devices; i++)
    free(binaries[i]);
  free(binaries);
ERROR_CLEAN_BINARIES_SIZES:
  free(binaries_sizes);
ERROR:
  return CL_OUT_OF_HOST_MEMORY;
}




int binaryFormat2Buffer(char *buffer, int sizeofBuffer, poclcc_global *binary_format){
  char *endofBuffer = buffer + sizeofBuffer;

  assert(poclcc_check_global(binary_format));
  memcpy(buffer, binary_format, sizeof(poclcc_global));
  buffer = (char *)(&(((poclcc_global *)buffer)->devices));
  assert(buffer < endofBuffer && "buffer is not a binaryformat");

  int i;
  for (i=0; i<binary_format->num_devices; i++){
    poclcc_device *device = &(binary_format->devices[i]);
    assert(poclcc_check_device(device));
    memcpy(buffer, device, sizeof(poclcc_device));
    buffer = (char*)(&(((poclcc_device *)buffer)->kernels));
    assert(buffer < endofBuffer && "buffer is not a binaryformat");

    int j;
    for (j=0; j<device->num_kernels; j++){
      poclcc_kernel *kernel = &(device->kernels[j]);

      *((uint32_t *)buffer) = kernel->sizeofKernelName;
      buffer += sizeof(uint32_t);
      assert(buffer < endofBuffer && "buffer is not a binaryformat");

      memcpy(buffer, kernel->kernel_name, kernel->sizeofKernelName);
      buffer += kernel->sizeofKernelName;
      assert(buffer < endofBuffer && "buffer is not a binaryformat");

      *((uint32_t *)buffer) = kernel->sizeofBinary;
      buffer += sizeof(uint32_t);
      assert(buffer < endofBuffer && "buffer is not a binaryformat");

      memcpy(buffer, kernel->binary, kernel->sizeofBinary);
      buffer += kernel->sizeofBinary;      
      assert(buffer < endofBuffer && "buffer is not a binaryformat");
    }
  }
  return CL_SUCCESS;
    /* (int)(endofBuffer - buffer) == 0?  */
    /* CL_SUCCESS: */
    /* (int)(endofBuffer - buffer);   */
}

int buffer2BinaryFormat(poclcc_global *binary_format, char *buffer, int sizeofBuffer){
  char *endofBuffer = buffer + sizeofBuffer;

  memcpy(binary_format, buffer, sizeof(poclcc_global));
  assert(poclcc_check_global(binary_format) && "check file identifier and version");
  buffer = (char *)(&(((poclcc_global *)buffer)->devices));
  assert(buffer < endofBuffer && "buffer is not a binaryformat");
  
  if ((binary_format->devices = malloc(binary_format->num_devices*sizeof(poclcc_device))) == NULL)
    goto ERROR;

  int i;
  for (i=0; i<binary_format->num_devices; i++){
    poclcc_device *device = &(binary_format->devices[i]);
    memcpy(device, buffer, sizeof(poclcc_device));
    buffer = (char *)&(((poclcc_device *)buffer)->kernels);
    assert(buffer < endofBuffer && "buffer is not a binaryformat");
    assert(poclcc_check_device(device));

    if ((device->kernels = malloc(device->num_kernels*sizeof(poclcc_kernel))) == NULL)
      goto ERROR_CLEAN_DEVICE;
   
    int j;
    for (j=0; j<device->num_kernels; j++){
      poclcc_kernel *kernel = &(device->kernels[j]);

      kernel->sizeofKernelName = *((uint32_t *)buffer);
      buffer += sizeof(uint32_t);
      assert(buffer < endofBuffer && "buffer is not a binaryformat");

      if ((kernel->kernel_name = malloc(kernel->sizeofKernelName*sizeof(char))) == NULL)
        goto ERROR_CLEAN_DEVICE_KERNEL;

      memcpy(kernel->kernel_name, buffer, kernel->sizeofKernelName);
      buffer += kernel->sizeofKernelName;
      assert(buffer < endofBuffer && "buffer is not a binaryformat");

      kernel->sizeofBinary = *((uint32_t *)buffer);
      buffer += sizeof(uint32_t);
      assert(buffer < endofBuffer && "buffer is not a binaryformat");
      
      if ((kernel->binary = malloc(kernel->sizeofBinary*sizeof(char))) == NULL)
        goto ERROR_CLEAN_DEVICE_KERNEL;

      memcpy(kernel->binary, buffer, kernel->sizeofBinary);
      buffer += kernel->sizeofBinary;      
      assert(buffer < endofBuffer && "buffer is not a binaryformat");

    }
  }
  return CL_SUCCESS;
    /* (int)(endofBuffer - buffer) == 0?  */
    /* CL_SUCCESS: */
    /* (int)(endofBuffer - buffer);   */

ERROR_CLEAN_DEVICE_KERNEL:
  for (i=0; i<binary_format->num_devices; i++){
    poclcc_device *device = &(binary_format->devices[i]);
    int j;
    for (j=0; j<device->num_kernels; j++){
      poclcc_kernel *kernel = &(device->kernels[j]);
      free(kernel->kernel_name);
      free(kernel->binary);
    }
    free(device->kernels);
  }
ERROR_CLEAN_DEVICE:
  free(binary_format->devices);
ERROR:
  return CL_OUT_OF_HOST_MEMORY;
}



void poclcc_init_global(poclcc_global *globalcc, int num_devices, poclcc_device *devices){
  globalcc->num_devices = num_devices;
  globalcc->devices = devices;
  strncpy(globalcc->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  globalcc->version = POCLCC_VERSION;
}

void poclcc_init_device(poclcc_device *devicecc, cl_device_id device, 
                        int num_kernels, poclcc_kernel *kernels){
  strncpy(devicecc->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  devicecc->device_id = poclcc_get_device_id(device);
  devicecc->num_kernels = num_kernels;
  devicecc->kernels = kernels;
}

void poclcc_init_kernel(poclcc_kernel *kernelcc, char *kernel_name, int sizeofKernelName, 
                        unsigned char *binary, int sizeofBinary){
  kernelcc->sizeofKernelName = sizeofKernelName;
  kernelcc->kernel_name = kernel_name;
  kernelcc->sizeofBinary = sizeofBinary;
  kernelcc->binary = binary;
}

int LookForKernelBinary(poclcc_global *binary_format, cl_device_id device, char *kernel_name, 
                        char **binary, int *binary_size){

  int i;
  for (i=0; i<binary_format->num_devices; i++){
    poclcc_device *devicecc = (poclcc_device *)(&(binary_format->devices[i]));
    if (poclcc_check_device(devicecc) && poclcc_check_device_id(device, devicecc)){
      int j;
      for (j=0; j<devicecc->num_kernels; j++){
        poclcc_kernel *kernel = &(devicecc->kernels[i]);
        if (!strncmp(kernel->kernel_name, kernel_name, kernel->sizeofKernelName)){
          int sizeofBinary = kernel->sizeofBinary;
          if ((*binary = malloc(sizeofBinary)) == NULL)
            return CL_OUT_OF_HOST_MEMORY;
          memcpy(*binary, kernel->binary, sizeofBinary);
          *binary_size = kernel->sizeofBinary;
          return CL_SUCCESS;
        }
      }
    }
  }
  return CL_INVALID_PROGRAM_EXECUTABLE;
}
