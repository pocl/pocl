#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clExtractBinaryFormat)(void *input_buffer,
                              cl_uint input_buffer_size,
                              size_t **lengths,
                              unsigned char ***binaries){

  const char *poclcc_string_id = POCLCC_STRING_ID;
  const cl_uint poclcc_version = POCLCC_VERSION;
  cl_int errcode = CL_SUCCESS;
  unsigned sizeof_string_id = strlen(poclcc_string_id);
  unsigned header_size = sizeof_string_id + sizeof(cl_uint);
  POCL_RETURN_ERROR_COND(input_buffer_size < header_size, 
                         CL_INVALID_BUFFER_SIZE);
    
  POCL_RETURN_ERROR_COND(
    strncmp(poclcc_string_id, input_buffer, sizeof_string_id), 
    CL_INVALID_BINARY);

  POCL_RETURN_ERROR_COND(
    poclcc_version != *((cl_uint*)(&((char*)input_buffer)[sizeof_string_id])),
    CL_INVALID_BINARY);

  unsigned num_binaries = 0;
  unsigned i = header_size;
  while (i < input_buffer_size){
    num_binaries++;
    i += *((cl_uint*)(&((char*)input_buffer)[i])) + sizeof(cl_uint);
  }
  POCL_RETURN_ERROR_COND(i != input_buffer_size, 
                         CL_INVALID_BUFFER_SIZE | CL_INVALID_BINARY);

  size_t *binaries_sizes;
  unsigned char **binaries_tab;
  POCL_RETURN_ERROR_COND(
    (binaries_sizes = malloc(num_binaries*sizeof(size_t))) == NULL, 
    CL_OUT_OF_HOST_MEMORY);
  
  if ((binaries_tab = malloc(num_binaries*sizeof(unsigned char*))) == NULL){
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR_CLEAN_BINARIES_SIZES;
  }

  char *buffer = (char *)input_buffer;
  buffer += header_size;
  for (i=0; i<num_binaries; i++){
    binaries_sizes[i] = *((uint32_t *)buffer);
    buffer += sizeof(uint32_t);

    if ((binaries_tab[i] = malloc(binaries_sizes[i]*sizeof(unsigned char))) 
        == NULL){
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

    memcpy(binaries_tab[i], buffer, binaries_sizes[i]);

    buffer += binaries_sizes[i];
  }

  *lengths = binaries_sizes;
  *binaries = binaries_tab;
  
  return errcode;

ERROR:
  for (i=0; i<num_binaries; i++)
    POCL_MEM_FREE(binaries_tab[i]);
  POCL_MEM_FREE(binaries_tab);
ERROR_CLEAN_BINARIES_SIZES:
  POCL_MEM_FREE(binaries_sizes);
  return errcode;

}
POsym(clExtractBinaryFormat)
