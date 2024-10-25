/* pocl_dkb_khr_dnn_utils_shared.h - collection of neural network util DBKs.

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

#include "pocl_dbk_khr_dnn_utils_shared.h"

int
pocl_validate_dnn_utils_attrs (BuiltinKernelId kernel_id,
                               const void *kernel_attributes)
{
  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_DNN_NMS:
      {
        cl_dbk_attributes_exp_dnn_nms *attrs
          = (cl_dbk_attributes_exp_dnn_nms *)kernel_attributes;
        POCL_RETURN_ERROR_ON (
          (attrs->nms_threshold > 1.0f || attrs->nms_threshold < 0.0f),
          CL_INVALID_DBK_ATTRIBUTE, "nms_threshold not between [0.0, 1.0].\n");
        POCL_RETURN_ERROR_ON (
          (attrs->score_threshold > 1.0f || attrs->score_threshold < 0.0f),
          CL_INVALID_DBK_ATTRIBUTE,
          "score_threshold not between [0.0, 1.0].\n");
        POCL_RETURN_ERROR_ON ((attrs->top_k < 0), CL_INVALID_DBK_ATTRIBUTE,
                              "top_k not larger than zero.\n");
        return CL_SUCCESS;
      }
    default:
      POCL_ABORT (
        "pocl_validate_dnn_utils_attrs called with wrong kernel_id.\n");
    }
}

int
pocl_release_dnn_utils_attrs (BuiltinKernelId kernel_id,
                              void *kernel_attributes)
{
  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_DNN_NMS:
      {
        POCL_MEM_FREE (kernel_attributes);
        return CL_SUCCESS;
      }
    default:
      POCL_ABORT ("pocl_release_dnn_utils_attrs called with "
                  "wrong kernel_id.\n");
    }
}

void *
pocl_copy_dnn_utils_attrs (BuiltinKernelId kernel_id,
                           const void *kernel_attributes)
{
  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_DNN_NMS:
      {
        void *ret = malloc (sizeof (cl_dbk_attributes_exp_dnn_nms));
        memcpy (ret, kernel_attributes,
                sizeof (cl_dbk_attributes_exp_dnn_nms));
        return ret;
      }
    default:
      POCL_ABORT ("pocl_copy_dnn_utils_attrs called with "
                  "wrong kernel_id.\n");
    }
}