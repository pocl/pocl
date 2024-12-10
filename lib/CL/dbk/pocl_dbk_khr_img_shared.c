/* pocl_dbk_khr_img_shared.c - generic color convert defined builtin kernel
   functions.

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
#include "pocl_dbk_khr_img_shared.h"

int
pocl_validate_img_attrs (cl_dbk_id_exp kernel_id,
                         const void *kernel_attributes)
{
  switch (kernel_id)
    {
    case CL_DBK_IMG_COLOR_CONVERT_EXP:
      {
        cl_dbk_attributes_img_color_convert_exp *attrs
          = (cl_dbk_attributes_img_color_convert_exp *)kernel_attributes;
        pocl_image_attr_t input_attr = attrs->input_image;
        pocl_image_attr_t output_attr = attrs->output_image;

        POCL_RETURN_ERROR_ON ((input_attr.format != POCL_DF_IMAGE_NV12
                               || output_attr.format != POCL_DF_IMAGE_RGB),
                              CL_DBK_INVALID_ATTRIBUTE_EXP,
                              "other color conversions than nv12->rgb have "
                              "not been implemented yet.\n");

        POCL_RETURN_ERROR_ON (
          (input_attr.color_space != POCL_COLOR_SPACE_BT709
           || output_attr.color_space != POCL_COLOR_SPACE_BT709),
          CL_DBK_INVALID_ATTRIBUTE_EXP,
          "other color spaces that BT709 have not been implemented yet.\n");

        POCL_RETURN_ERROR_ON (
          (input_attr.channel_range != POCL_CHANNEL_RANGE_FULL
           || output_attr.channel_range != POCL_CHANNEL_RANGE_FULL),
          CL_DBK_INVALID_ATTRIBUTE_EXP,
          "other than channel ranges than POCL_CHANNEL_RANGE_FULL have not "
          "been implemented yet.\n");

        if ((input_attr.height == 0 || input_attr.width == 0)
            && (output_attr.height == 0 || output_attr.width == 0))
          POCL_MSG_WARN (
            "no image attribute contains both width and height populated \n");

        return CL_SUCCESS;
      }
    default:
      POCL_ABORT ("pocl_validate_img_attrs called with wrong kernel_id\n");
    }
}

int
pocl_release_img_attrs (cl_dbk_id_exp kernel_id, void *kernel_attributes)
{
  switch (kernel_id)
    {
    case CL_DBK_IMG_COLOR_CONVERT_EXP:
      {
        POCL_MEM_FREE (kernel_attributes);
        return CL_SUCCESS;
      }
    default:
      POCL_ABORT ("pocl_release_img_attrs called with "
                  "wrong kernel_id\n");
    }
}

void *
pocl_copy_img_attrs (cl_dbk_id_exp kernel_id, const void *kernel_attributes)
{
  switch (kernel_id)
    {

    case CL_DBK_IMG_COLOR_CONVERT_EXP:
      {
        void *ret = malloc (sizeof (cl_dbk_attributes_img_color_convert_exp));
        memcpy (ret, kernel_attributes,
                sizeof (cl_dbk_attributes_img_color_convert_exp));
        return ret;
      }
    default:
      POCL_ABORT ("pocl_copy_img_attrs called with "
                  "wrong kernel_id\n");
    }
}