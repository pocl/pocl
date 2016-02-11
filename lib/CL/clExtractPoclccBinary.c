#include "pocl_util.h"
#include "poclcc_binary.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clExtractPoclccBinary)(void *input_buffer,
                              cl_uint input_buffer_size,
                              size_t **lengths,
                              unsigned char ***binaries){
  cl_int errcode = CL_SUCCESS;

  poclcc_global poclcc;
  POCL_RETURN_ERROR_COND(
    (errcode=poclcc_buffer2BinaryFormat(&poclcc, 
                                        input_buffer, 
                                        input_buffer_size)) != CL_SUCCESS,
    errcode);

  POCL_RETURN_ERROR_COND(
    (errcode=poclcc_binaryFormat2ProgramInfos(binaries, lengths, &poclcc)) != CL_SUCCESS,
    errcode);

  poclcc_free(&poclcc);

  return errcode;
}
POsym(clExtractPoclccBinary)
