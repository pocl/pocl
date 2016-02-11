#include "poclcc_binary.h"

void poclcc_free(poclcc_global *binary_format){
  if (binary_format != NULL)
    if (binary_format->devices != NULL)
      {
        int i;
        for (i=0; i<binary_format->num_devices; i++)
          poclcc_device_free(&(binary_format->devices[i]));
        POCL_MEM_FREE(binary_format->devices);
      }
}

void poclcc_device_free(poclcc_device *device){
  if (device != NULL)
    if (device->kernels != NULL)
      {
        int j;
        for (j=0; j<device->num_kernels; j++)
          poclcc_kernel_free(&(device->kernels[j]));
        POCL_MEM_FREE(device->kernels);
      }
}

void poclcc_kernel_free(poclcc_kernel *kernel){
  if (kernel != NULL)
    {
      POCL_MEM_FREE(kernel->kernel_name);
      POCL_MEM_FREE(kernel->binary);
      POCL_MEM_FREE(kernel->dyn_arguments);
      POCL_MEM_FREE(kernel->arg_info);
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

#define FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x100000001b3)
uint64_t poclcc_get_device_id(cl_device_id device){
  //FNV-1A with vendor_id, llvm_target_triplet and llvm_cpu
  uint64_t result = FNV_OFFSET;
  const char *llvm_tt = device->llvm_target_triplet;
  const char *llvm_cpu = device->llvm_cpu;

  result *= FNV_PRIME;
  result ^= device->vendor_id;
  int i, length = strlen(llvm_tt);
  for (i=0; i<length; i++)
    {
      result *= FNV_PRIME;
      result ^= llvm_tt[i];
    }
  length = strlen(llvm_cpu);
  for (i=0; i<length; i++)
    {
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


int poclcc_sizeofKernel(poclcc_kernel *kernel){
  return sizeof(poclcc_kernel) 
    + kernel->sizeofKernelName + kernel->sizeofBinary
    - sizeof(kernel->kernel_name) - sizeof(kernel->binary) 
    + (kernel->num_args + kernel->num_locals) * sizeof(struct pocl_argument)
    + kernel->num_args * sizeof(struct pocl_argument_info)
    - sizeof (kernel->dyn_arguments) - sizeof(kernel->arg_info);
}

int poclcc_sizeofDevice(poclcc_device *device){
  int size = sizeof(poclcc_device) - sizeof(device->kernels);
  int i;
  for (i=0; i<device->num_kernels; i++)
    size += poclcc_sizeofKernel(&(device->kernels[i]));

  return size;
}

int poclcc_sizeofGlobal(poclcc_global *binary_format){
  int size = 0;
  if (binary_format == NULL)
    return -1;
    
  size += sizeof(poclcc_global) - sizeof(poclcc_device *);
  int i;
  for (i=0; i<binary_format->num_devices; i++)
    size += poclcc_sizeofDevice(&(binary_format->devices[i]));
  
  return size;
}

int poclcc_sizeofGlobalFromBinariesSizes(size_t *binaries_sizes, int num_devices){
  if (binaries_sizes == NULL)
    return -1;

  int i;
  int size = 0;
  for (i=0; i<num_devices; i++)
    size += binaries_sizes[i];

  size += sizeof(poclcc_global) - sizeof(poclcc_device *);
  return size;
}


void copyKernel2Buffer(poclcc_kernel *kernel, char **buf){
  char *buffer = *buf;
  int sizeofKernelName = kernel->sizeofKernelName;
  int sizeofBinary = kernel->sizeofBinary;
  int sizeofDynArgs = (kernel->num_args + kernel->num_locals) 
    * sizeof(struct pocl_argument);
  int sizeofArgInfo = kernel->num_args * sizeof(struct pocl_argument_info);

  memcpy(buffer, kernel, sizeof(poclcc_kernel));
  buffer = (unsigned char *)(&(((poclcc_kernel *)buffer)->kernel_name));

  memcpy(buffer, kernel->kernel_name, sizeofKernelName);
  buffer += sizeofKernelName;

  memcpy(buffer, kernel->dyn_arguments, sizeofDynArgs);
  buffer += sizeofDynArgs;

  memcpy(buffer, kernel->arg_info, sizeofArgInfo);
  buffer += sizeofArgInfo;

  memcpy(buffer, kernel->binary, sizeofBinary);
  buffer += sizeofBinary;
  *buf = buffer;
}
int copyBuffer2Kernel(char **buf, poclcc_kernel *kernel){
  char *kernel_src = *buf;
  poclcc_kernel *kernel_tmp_src = (poclcc_kernel *)kernel_src;

  memcpy(kernel, kernel_tmp_src, sizeof(poclcc_kernel));

  int sizeofKernelName = kernel_tmp_src->sizeofKernelName;
  int sizeofBinary = kernel_tmp_src->sizeofBinary;
  int sizeofDynArgs = 
    (kernel_tmp_src->num_args + kernel_tmp_src->num_locals) 
    * sizeof(struct pocl_argument);
  int sizeofArgInfo = 
    kernel_tmp_src->num_args * sizeof(struct pocl_argument_info);

  kernel_src = (char *)(&(kernel_tmp_src->kernel_name));

  if ((kernel->kernel_name = malloc(sizeofKernelName)) == NULL)
    goto ERROR;
  memcpy(kernel->kernel_name, kernel_src, sizeofKernelName);
  kernel_src += sizeofKernelName;

  if ((kernel->dyn_arguments = malloc(sizeofDynArgs)) == NULL)
    goto ERROR;
  memcpy(kernel->dyn_arguments, kernel_src, sizeofDynArgs);
  kernel_src += sizeofDynArgs;

  if ((kernel->arg_info = malloc(sizeofArgInfo)) == NULL)
    goto ERROR;
  memcpy(kernel->arg_info, kernel_src, sizeofArgInfo);
  kernel_src += sizeofArgInfo;

  if ((kernel->binary = malloc(sizeofBinary)) == NULL)
    goto ERROR;
  memcpy(kernel->binary, kernel_src, sizeofBinary);
  kernel_src += sizeofBinary;

  *buf = kernel_src;
  return CL_SUCCESS;
ERROR:
  poclcc_kernel_free(kernel);
  return CL_OUT_OF_HOST_MEMORY;
}


cl_int poclcc_programInfos2BinaryFormat(poclcc_global *binary_format, unsigned char **binaries,
                                 unsigned num_devices){
  strncpy(binary_format->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  binary_format->version = POCLCC_VERSION;
  binary_format->num_devices = num_devices;
  if ((binary_format->devices = malloc(num_devices*sizeof(poclcc_device))) == NULL)
    goto ERROR;

  int i;
  for (i=0; i<num_devices; i++)
    {
      poclcc_device *device_dst = &(binary_format->devices[i]);
      poclcc_device *device_src = (poclcc_device *)(binaries[i]);
      memcpy(device_dst, device_src, sizeof(poclcc_device));    
      assert(poclcc_check_device(device_dst));
      
      int num_kernels = device_dst->num_kernels;
      if ((device_dst->kernels = malloc(num_kernels*sizeof(poclcc_kernel))) == NULL)
        {
          poclcc_free(binary_format);
          goto ERROR;
        }
      int j;
      char *kernel_src = (char *)(&(device_src->kernels));
      for (j=0; j<num_kernels; j++)
        {
          poclcc_kernel *kernel_dst = &(device_dst->kernels[i]);
          if (copyBuffer2Kernel(&kernel_src, kernel_dst) != CL_SUCCESS)
            goto ERROR;
        }
    }
  return CL_SUCCESS;
ERROR:
  poclcc_free(binary_format);
  return CL_OUT_OF_HOST_MEMORY;
}

cl_int poclcc_binaryFormat2ProgramInfos(unsigned char ***binaries, size_t **binaries_sizes, 
                                 poclcc_global *binary_format){
  assert(poclcc_check_global(binary_format));
  int num_devices = binary_format->num_devices;

  if ((*binaries_sizes = malloc(sizeof(size_t)*num_devices)) == NULL)
    goto ERROR;
  
  if ((*binaries = malloc(sizeof(unsigned char*)*num_devices)) == NULL)
    goto ERROR_CLEAN_BINARIES_SIZES;
  
  int i;
  for (i=0; i<num_devices; i++)
    {
      char *devicecc_binary;
      poclcc_device *device = &(binary_format->devices[i]);
      int sizeofDevicecc = poclcc_sizeofDevice(device);
      if ((devicecc_binary = malloc(sizeofDevicecc)) == NULL)
        goto ERROR_CLEAN_BINARIES_ALL;
      (*binaries)[i] = devicecc_binary;
      (*binaries_sizes)[i] = sizeofDevicecc;
      
      memcpy(devicecc_binary, device, sizeof(poclcc_device));
      devicecc_binary += sizeof(poclcc_device) - sizeof(poclcc_kernel *);
      
      int j;
      for (j=0; j<device->num_kernels; j++)
        {
          poclcc_kernel *kernel = &(device->kernels[j]);
          copyKernel2Buffer(kernel, &devicecc_binary);
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



int poclcc_binaryFormat2Buffer(char *buffer, int sizeofBuffer, poclcc_global *binary_format){
  char *endofBuffer = buffer + sizeofBuffer;

  assert(poclcc_check_global(binary_format));
  memcpy(buffer, binary_format, sizeof(poclcc_global));
  buffer = (char *)(&(((poclcc_global *)buffer)->devices));
  assert(buffer < endofBuffer);

  int i;
  for (i=0; i<binary_format->num_devices; i++)
    {
      poclcc_device *device = &(binary_format->devices[i]);
      assert(poclcc_check_device(device));
      memcpy(buffer, device, sizeof(poclcc_device));
      buffer = (char*)(&(((poclcc_device *)buffer)->kernels));
      assert(buffer < endofBuffer);
      
      int j;
      for (j=0; j<device->num_kernels; j++)
        {
          poclcc_kernel *kernel = &(device->kernels[j]);
          copyKernel2Buffer(kernel, &buffer);
          assert(buffer <= endofBuffer);
        }
    }
  return CL_SUCCESS;
}

int poclcc_buffer2BinaryFormat(poclcc_global *binary_format, char *buffer, int sizeofBuffer){
  char *endofBuffer = buffer + sizeofBuffer;

  memcpy(binary_format, buffer, sizeof(poclcc_global));
  assert(poclcc_check_global(binary_format));
  buffer = (char *)(&(((poclcc_global *)buffer)->devices));
  assert(buffer < endofBuffer);
  
  if ((binary_format->devices = malloc(binary_format->num_devices*sizeof(poclcc_device))) == NULL)
    goto ERROR;

  int i;
  for (i=0; i<binary_format->num_devices; i++)
    {
      poclcc_device *device = &(binary_format->devices[i]);
      memcpy(device, buffer, sizeof(poclcc_device));
      buffer = (char *)&(((poclcc_device *)buffer)->kernels);
      assert(buffer < endofBuffer);
      assert(poclcc_check_device(device));
      
      if ((device->kernels = malloc(device->num_kernels*sizeof(poclcc_kernel))) == NULL)
        goto ERROR;
      
      int j;
      for (j=0; j<device->num_kernels; j++)
        {
          poclcc_kernel *kernel = &(device->kernels[j]);
          if (copyBuffer2Kernel(&buffer, kernel) != CL_SUCCESS)
            goto ERROR;
          assert(buffer <= endofBuffer);
        }
    }
  return CL_SUCCESS;
  
ERROR:
  poclcc_free(binary_format);
  return CL_OUT_OF_HOST_MEMORY;
}


poclcc_kernel *LookForKernelcc(poclcc_global *binary_format, cl_device_id device, const char *kernel_name){
  int i;
  for (i=0; i<binary_format->num_devices; i++)
    {
      poclcc_device *devicecc = (poclcc_device *)(&(binary_format->devices[i]));
      if (poclcc_check_device(devicecc) && poclcc_check_device_id(device, devicecc))
        {
          int j;
          for (j=0; j<devicecc->num_kernels; j++)
            {
              poclcc_kernel *kernel = &(devicecc->kernels[i]);
              if (!strncmp(kernel->kernel_name, kernel_name, kernel->sizeofKernelName))
                return kernel;
            }
        }
    }
  return NULL;
}

cl_int poclcc_binaryFormat2ClKernel(poclcc_global *binary_format, const char *kernel_name,
                             cl_kernel kernel, cl_device_id device){
  poclcc_kernel *kernelcc = LookForKernelcc(binary_format, device, kernel_name);
  if (kernelcc == NULL)
    return CL_INVALID_PROGRAM_EXECUTABLE;

  kernel->num_args = kernelcc->num_args;
  kernel->num_locals = kernelcc->num_locals;

  int sizeofDynArgs = (kernel->num_args + kernel->num_locals) * sizeof(struct pocl_argument);
  int sizeofArgInfo = kernel->num_args * sizeof(struct pocl_argument_info);

  if ((kernel->dyn_arguments = malloc(sizeofDynArgs)) == NULL)
    goto ERROR;
  memcpy(kernel->dyn_arguments, kernelcc->dyn_arguments, sizeofDynArgs);

  if ((kernel->arg_info = malloc(sizeofArgInfo)) == NULL)
    goto ERROR;
  memcpy(kernel->arg_info, kernelcc->arg_info, sizeofArgInfo);

  if ((kernel->reqd_wg_size = malloc(3*sizeof(int))) == NULL)
    goto ERROR; 

  return CL_SUCCESS;
ERROR:
  free(kernel->reqd_wg_size);
  free(kernel->dyn_arguments);
  free(kernel->arg_info);
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
                        unsigned char *binary, int sizeofBinary, int num_args, int num_locals,
                        struct pocl_argument *dyn_arguments, struct pocl_argument_info *arg_info){
  kernelcc->sizeofKernelName = sizeofKernelName;
  kernelcc->kernel_name = kernel_name;
  kernelcc->sizeofBinary = sizeofBinary;
  kernelcc->binary = binary;
  kernelcc->num_args = num_args;
  kernelcc->num_locals = num_locals;
  kernelcc->dyn_arguments = dyn_arguments;
  kernelcc->arg_info = arg_info;
}


int poclcc_LookForKernelBinary(poclcc_global *binary_format, cl_device_id device, const char *kernel_name, 
                        char **binary, int *binary_size){
  poclcc_kernel *kernel = LookForKernelcc(binary_format, device, kernel_name);
  if (kernel == NULL)
    return CL_INVALID_PROGRAM_EXECUTABLE;
    
  int sizeofBinary = kernel->sizeofBinary;
  if ((*binary = malloc(sizeofBinary)) == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  memcpy(*binary, kernel->binary, sizeofBinary);
  *binary_size = kernel->sizeofBinary;

  return CL_SUCCESS;
}
