#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clExportBinaryFormat)(cl_program program,
                             void **binary_format,
                             cl_uint *binary_size){
  const char *poclcc_string_id = POCLCC_STRING_ID;
  const cl_uint poclcc_version = POCLCC_VERSION;
  cl_int num_devices;
  size_t size_ret;
  int i,
    sizeof_binaries = 0,
    sizeof_string_id = strlen(poclcc_string_id)*sizeof(char);
  size_t *binaries_sizes;
  unsigned char **binaries;
  cl_int errcode = CL_SUCCESS;

  POCL_RETURN_ERROR_COND(program == NULL, CL_INVALID_PROGRAM);

  errcode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, 
                             sizeof(cl_int), &num_devices, &size_ret);
  POCL_RETURN_ERROR_COND(errcode != 0, errcode);

  POCL_RETURN_ERROR_COND((binaries_sizes = malloc(num_devices*sizeof(size_t)))
                         == NULL, 
                         CL_OUT_OF_HOST_MEMORY);

  errcode = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 
                             num_devices*sizeof(size_t), 
                             binaries_sizes, &size_ret);
  if (errcode != 0) goto ERROR_CLEAN_BINARIES_SIZES;

  if ((binaries = malloc(num_devices*sizeof(unsigned char*))) == NULL){
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR_CLEAN_BINARIES_SIZES;
  }
  
  for (i=0; i<num_devices; i++){
    if ( (binaries[i] = malloc(binaries_sizes[i]*sizeof(unsigned char))) 
        == NULL ){
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
    sizeof_binaries += binaries_sizes[i];
  }

  errcode = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 
                             num_devices*sizeof(unsigned char*), 
                             binaries, &size_ret);
  if (errcode != 0) goto ERROR;

  *binary_size = sizeof_string_id 
    + sizeof(cl_uint) 
    + num_devices*sizeof(size_t) 
    + sizeof_binaries;
  if ((*binary_format = malloc(*binary_size)) == NULL){
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  void *bf = *binary_format;
  strcpy(bf, poclcc_string_id);
  bf += sizeof_string_id;

  memcpy(bf, &poclcc_version, sizeof(cl_uint));
  bf += sizeof(cl_uint);

  for (i=0; i<num_devices; i++){
    memcpy(bf, &binaries_sizes[i], sizeof(size_t));
    bf += sizeof(size_t);
    memcpy(bf, binaries[i], binaries_sizes[i]*sizeof(char));
    bf += binaries_sizes[i]*sizeof(char);
  }

  return errcode;

ERROR:
  for (--i; i>=0; i--)
    POCL_MEM_FREE(binaries[i]);
  POCL_MEM_FREE(binaries);
ERROR_CLEAN_BINARIES_SIZES:
  POCL_MEM_FREE(binaries_sizes);
  return errcode;

}
POsym(clExportBinaryFormat)
