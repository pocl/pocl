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
pocl_validate_khr_jpeg (cl_dbk_id_exp kernel_id, const void *kernel_attributes)
{

  switch (kernel_id)
    {
    case CL_DBK_JPEG_ENCODE_EXP:
      {
        cl_dbk_attributes_jpeg_encode_exp *attrs
          = (cl_dbk_attributes_jpeg_encode_exp *)kernel_attributes;
        POCL_RETURN_ERROR_ON ((1 > attrs->height || attrs->height > 65535),
                              CL_DBK_INVALID_ATTRIBUTE_EXP,
                              "Height not between (0, 65535].\n");
        POCL_RETURN_ERROR_ON ((1 > attrs->width || attrs->width > 65535),
                              CL_DBK_INVALID_ATTRIBUTE_EXP,
                              "Width not between (0, 65535].\n");
        POCL_RETURN_ERROR_ON ((0 > attrs->quality || attrs->quality > 100),
                              CL_DBK_INVALID_ATTRIBUTE_EXP,
                              "Quality not between [0, 100].\n");
        return CL_SUCCESS;
      }
    case CL_DBK_JPEG_DECODE_EXP:
      {
        POCL_RETURN_ERROR_ON (kernel_attributes != NULL,
                              CL_DBK_INVALID_ATTRIBUTE_EXP,
                              "decode attributes should be null. \n");
        return CL_SUCCESS;
      }

    default:
      POCL_MSG_ERR ("pocl_validate_khr_jpeg called with wrong kernel_id.\n");
      return CL_FAILED;
    }
}

void *
pocl_copy_dbk_attributes_khr_jpeg (cl_dbk_id_exp kernel_id,
                                   const void *kernel_attributes)
{

  switch (kernel_id)
    {
    case CL_DBK_JPEG_ENCODE_EXP:
      {
        void *ret = malloc (sizeof (cl_dbk_attributes_jpeg_encode_exp));
        memcpy (ret, kernel_attributes,
                sizeof (cl_dbk_attributes_jpeg_encode_exp));
        return ret;
      }
    case CL_DBK_JPEG_DECODE_EXP:
      return NULL;
    default:
      POCL_MSG_ERR ("pocl_copy_dbk_attributes_khr_jpeg called with "
                    "wrong kernel_id.\n");
      return NULL;
    }
}

int
pocl_release_dbk_attributes_khr_jpeg (cl_dbk_id_exp kernel_id,
                                      void *kernel_attributes)
{

  switch (kernel_id)
    {
    case CL_DBK_JPEG_ENCODE_EXP:
      {
        POCL_MEM_FREE (kernel_attributes);
        return CL_SUCCESS;
      }
    case CL_DBK_JPEG_DECODE_EXP:
      {
        POCL_MEM_FREE (kernel_attributes);
        return CL_SUCCESS;
      }
    default:
      POCL_RETURN_ERROR (CL_DBK_INVALID_ID_EXP,
                         "pocl_copy_dbk_attributes_khr_jpeg called with "
                         "wrong kernel_id.\n");
    }
  assert (!"UNREACHABLE");
  return CL_DBK_INVALID_ID_EXP;
}
