/* pocl_dbk_khr_jpeg_shared.c - generic JPEG Defined Built-in Kernel functions.

   Copyright (c) 2024 Robin Bijl / Tampere University

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

#include "pocl_dbk_khr_jpeg_shared.h"

int
pocl_validate_khr_jpeg (BuiltinKernelId kernel_id,
                        const void *kernel_attributes)
{

  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        cl_dbk_attributes_exp_jpeg_encode *attrs
          = (cl_dbk_attributes_exp_jpeg_encode *)kernel_attributes;
        POCL_RETURN_ERROR_ON ((1 > attrs->height || attrs->height > 65535),
                              CL_INVALID_DBK_ATTRIBUTE,
                              "Height not between (0, 65535].\n");
        POCL_RETURN_ERROR_ON ((1 > attrs->width || attrs->width > 65535),
                              CL_INVALID_DBK_ATTRIBUTE,
                              "Width not between (0, 65535].\n");
        POCL_RETURN_ERROR_ON ((0 > attrs->quality || attrs->quality > 100),
                              CL_INVALID_DBK_ATTRIBUTE,
                              "Quality not between [0, 100].\n");
        return CL_SUCCESS;
      }
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      {
        POCL_RETURN_ERROR_ON (kernel_attributes != NULL,
                              CL_INVALID_DBK_ATTRIBUTE,
                              "decode attributes should be null. \n");
        return CL_SUCCESS;
      }

    default:
      POCL_MSG_ERR ("pocl_validate_khr_jpeg called with wrong kernel_id.\n");
      return CL_FAILED;
    }
}

void *
pocl_copy_dbk_attributes_khr_jpeg (BuiltinKernelId kernel_id,
                                   const void *kernel_attributes)
{

  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        void *ret = malloc (sizeof (cl_dbk_attributes_exp_jpeg_encode));
        memcpy (ret, kernel_attributes,
                sizeof (cl_dbk_attributes_exp_jpeg_encode));
        return ret;
      }
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      return NULL;
    default:
      POCL_MSG_ERR ("pocl_copy_dbk_attributes_khr_jpeg called with "
                    "wrong kernel_id.\n");
      return NULL;
    }
}

int
pocl_release_dbk_attributes_khr_jpeg (BuiltinKernelId kernel_id,
                                      void *kernel_attributes)
{

  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        POCL_MEM_FREE (kernel_attributes);
        return CL_SUCCESS;
      }
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      {
        POCL_MEM_FREE (kernel_attributes);
        return CL_SUCCESS;
      }
    default:
      POCL_RETURN_ERROR (CL_INVALID_DBK_ID,
                         "pocl_copy_dbk_attributes_khr_jpeg called with "
                         "wrong kernel_id.\n");
    }
}
