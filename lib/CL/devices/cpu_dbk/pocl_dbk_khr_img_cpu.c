/* pocl_dbk_khr_img_cpu.c - cpu implementation of image related dbks.

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

#include "pocl_dbk_khr_img_cpu.h"
#include "pocl_mem_management.h"

int
pocl_cpu_execute_dbk_exp_img_yuv2rgb (cl_program program,
                                      cl_kernel kernel,
                                      pocl_kernel_metadata_t *meta,
                                      cl_uint dev_i,
                                      struct pocl_argument *arguments)
{

  cl_device_id dev = program->devices[dev_i];
  cl_dbk_attributes_img_color_convert_exp *attrs = meta->builtin_kernel_attrs;
  unsigned mem_id = dev->global_mem_id;
  uint8_t *input = pocl_cpu_get_ptr (&arguments[0], mem_id);
  uint8_t *output = pocl_cpu_get_ptr (&arguments[1], mem_id);

  assert (attrs->input_image.format == POCL_DF_IMAGE_NV12
          && "other yuv formats not supported yet");
  assert (attrs->output_image.format == POCL_DF_IMAGE_RGB
          && "other rgb formats not supported yet");

  int width = attrs->input_image.width;
  int height = attrs->input_image.height;
  if (attrs->input_image.width == 0 || attrs->input_image.height == 0)
    {
      width = attrs->output_image.width;
      height = attrs->output_image.height;
    }
  size_t tot_pixels = (size_t)width * height;

  cl_mem input_mem = *(cl_mem *)(arguments[0].value);
  if (input_mem->size < tot_pixels * 3 / 2)
    {
      POCL_MSG_ERR ("pocl_cpu_execute_dbk_exp_img_yuv2rgb, "
                    "input memory is not of the correct size \n");
      assert (0);
    }

  cl_mem output_mem = *(cl_mem *)(arguments[1].value);
  if (output_mem->size < tot_pixels * 3)
    {
      POCL_MSG_ERR ("pocl_cpu_execute_dbk_exp_img_yuv2rgb, "
                    "output memory does not fit result \n");
      assert (0);
    }

  int pixel_index, uv_index;
  int y_value, u_value, v_value;
  int r, g, b;

  for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
        {

          uv_index = width * (y / 2) + (x / 2) * 2;
          pixel_index = y * width + x;

          y_value = input[pixel_index];
          /* convert from [0, 255] range to [-128, 127] range */
          u_value = input[tot_pixels + uv_index] - 128;
          v_value = input[tot_pixels + uv_index + 1] - 128;

          /* convert to rgb with BT.709 weights, useful references:
           * * https://registry.khronos.org/OpenVX/specs/1.3.1/html/OpenVX_Specification_1_3_1.html#group_vision_function_colorconvert
           * * https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
           */
          r = (int)(y_value + 1.5748f * v_value);
          g = (int)(y_value - (0.4681f * v_value) - (0.1873f * u_value));
          b = (int)(y_value + 1.8556f * u_value);

          /* clamp to uint8_t range */
          r = r > 255 ? 255 : (r < 0 ? 0 : r);
          g = g > 255 ? 255 : (g < 0 ? 0 : g);
          b = b > 255 ? 255 : (b < 0 ? 0 : b);

          /* write results back */
          output[pixel_index * 3] = r & 0xff;
          output[pixel_index * 3 + 1] = g & 0xff;
          output[pixel_index * 3 + 2] = b & 0xff;
        }
    }

  return CL_SUCCESS;
}
