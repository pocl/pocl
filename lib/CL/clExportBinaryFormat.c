#include "pocl_util.h"
#include "pocl_binary_format.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clExportBinaryFormat)(cl_program program,
                             void **binary_format,
                             cl_uint *binary_size){
  cl_int num_devices;
  size_t size_ret;
  int i;
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
  }

  errcode = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 
                             num_devices*sizeof(unsigned char*), 
                             binaries, &size_ret);
  if (errcode != 0) goto ERROR;

  *binary_size = sizeofPoclccGlobalFromBinariesSizes(binaries_sizes, 
                                                     num_devices);
  if ((*binary_format = malloc(*binary_size)) == NULL){
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }
  
  poclcc_global poclcc;
  POCL_GOTO_ERROR_COND(programInfos2BinaryFormat(&poclcc, binaries, num_devices)
                       != CL_SUCCESS, 
                       CL_OUT_OF_HOST_MEMORY);
  POCL_GOTO_ERROR_COND(
    (errcode=binaryFormat2Buffer(*binary_format, *binary_size, &poclcc)) != CL_SUCCESS,
    errcode);

  poclcc_free(&poclcc);

  return errcode;

ERROR:
  for (i=0; i<num_devices; i++)
    POCL_MEM_FREE(binaries[i]);
  POCL_MEM_FREE(binaries);
ERROR_CLEAN_BINARIES_SIZES:
  POCL_MEM_FREE(binaries_sizes);
  return errcode;

}
POsym(clExportBinaryFormat)
