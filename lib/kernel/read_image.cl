/* OpenCL built-in library: read_image()

   Copyright (c) 2013 Ville Korhonen
   Copyright (c) 2014 Felix Bytow
   Copyright (c) 2015 Matias Koskela

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

/* NOTE: this file is NOT a generic implementation; it works with vectors
   in a lot of places and requires that either device supports unaligned
   vector operations, or that memory backing the images is properly aligned.
   The maximum required alignment is 16bytes (4channels * 32bit color)

   Not all CPUs support unaligned vector operations, but the pthread / basic
   drivers allocate properly aligned memory for backing buffers; therefore
   this should work for everything supported by pthread / basic.
*/

#include "templates.h"
#include "pocl_image_rw_utils.h"

static uint4
map_channels (uint4 color, int order)
{
  switch (order)
    {
    case CL_ARGB:
      return color.yzwx;
    case CL_BGRA:
      return color.zyxw;
    case CL_RGBA:
    default:
      return color;
    }
}

/*************************************************************************/

/* only for CL_FLOAT, CL_SNORM_INT8, CL_UNORM_INT8,
 * CL_SNORM_INT16, CL_UNORM_INT16 channel types */
static float4
get_float4_pixel (void *data, size_t base_index, int type)
{
  if (type == CL_FLOAT)
    return ((float4 *)data)[base_index];
  if (type == CL_HALF_FLOAT)
    {
#if !defined(LLVM_OLDER_THAN_3_9) && __has_builtin(__builtin_convertvector)
      typedef float  vector4float  __attribute__((__vector_size__(16)));
      typedef half  vector4half  __attribute__((__vector_size__(8)));
      vector4half vh = ((vector4half *)data)[base_index];
      vector4float vf = __builtin_convertvector(vh, vector4float);
      return vf;
#else
      __builtin_trap();
#endif
    }
  const float4 one_127th = (float4) (1.0f / 127.0f);
  const float4 one_32767th = (float4) (1.0f / 32767.0f);
  const float4 one_255th = ((float4) (1.0f / (float)UCHAR_MAX));
  const float4 one_65535th = ((float4) (1.0f / (float)USHRT_MAX));
  if (type == CL_SNORM_INT8)
    {
      /*  <I*_MIN, I*_MAX> to <-1.0, 1.0> */
      int4 color = convert_int4 (((char4 *)data)[base_index]);
      float4 colorf = convert_float4 (color);
      return max ((float4) (-1.0f), (one_127th * colorf));
    }
  if (type == CL_SNORM_INT16)
    {
      int4 color = convert_int4 (((short4 *)data)[base_index]);
      float4 colorf = convert_float4 (color);
      return max ((float4) (-1.0f), (one_32767th * colorf));
    }
  if (type == CL_UNORM_INT8)
    {
      /* <0, I*_MAX> to <0.0, 1.0> */
      return convert_float4 (((uchar4 *)data)[base_index]) * one_255th;
    }
  if (type == CL_UNORM_INT16)
    {
      /* <0, I*_MAX> to <0.0, 1.0> */
      return convert_float4 (((ushort4 *)data)[base_index]) * one_65535th;
    }
  return (float4) (123.0f);
}

/* only for CL_FLOAT, CL_SNORM_INT8, CL_UNORM_INT8,
 * CL_SNORM_INT16, CL_UNORM_INT16 channel types */
static float
get_float_pixel (void *data, size_t base_index, int type)
{
  if (type == CL_FLOAT)
    return ((float *)data)[base_index];
  const float one_127th = (float)(1.0f / 127.0f);
  const float one_32767th = (float)(1.0f / 32767.0f);
  const float one_255th = ((float)(1.0f / (float)UCHAR_MAX));
  const float one_65535th = ((float)(1.0f / (float)USHRT_MAX));
  if (type == CL_SNORM_INT8)
    {
      /*  <I*_MIN, I*_MAX> to <-1.0, 1.0> */
      char color = ((char *)data)[base_index];
      float colorf = convert_float (color);
      return max ((-1.0f), (one_127th * colorf));
    }
  if (type == CL_SNORM_INT16)
    {
      short color = ((short *)data)[base_index];
      float colorf = convert_float (color);
      return max ((-1.0f), (one_32767th * colorf));
    }
  if (type == CL_UNORM_INT8)
    {
      /* <0, I*_MAX> to <0.0, 1.0> */
      return convert_float (((uchar *)data)[base_index]) * one_255th;
    }
  if (type == CL_UNORM_INT16)
    {
      return convert_float (((ushort *)data)[base_index]) * one_65535th;
    }

  return 234.0f;
}

/*************************************************************************/

#define BORDER_COLOR (0)
#define BORDER_COLOR_F (0.0f)

/* for use inside filter functions
 * no channel mapping
 * no pointers to img metadata */
static uint4
pocl_read_pixel_fast_ui (int4 coord, int width, int height, int depth,
                         size_t row_pitch, size_t slice_pitch, int order,
                         int elem_size, void *data)
{
  uint4 color;
  size_t base_index
      = coord.x + (coord.y * row_pitch) + (coord.z * slice_pitch);

  if ((coord.x >= width || coord.x < 0)
      || ((height != 0) && (coord.y >= height || coord.y < 0))
      || ((depth != 0) && (coord.z >= depth || coord.z < 0)))
    {
      return (uint4)BORDER_COLOR;
      /* if out of bounds, return BORDER COLOR:
       * since pocl's basic/pthread device only
       * supports CL_A + CL_{RGBA combos},
       * the border color is always zeroes. */
    }

  if (order == CL_A)
    {
      color = (uint4)0;
      if (elem_size == 1)
        color.w = ((uchar *)data)[base_index];
      else if (elem_size == 2)
        color.w = ((ushort *)data)[base_index];
      else if (elem_size == 4)
        color.w = ((uint *)data)[base_index];
      return color;
    }

  if (elem_size == 1)
    {
      return convert_uint4 (((uchar4 *)data)[base_index]);
    }
  else if (elem_size == 2)
    {
      return convert_uint4 (((ushort4 *)data)[base_index]);
    }
  else if (elem_size == 4)
    {
      return ((uint4 *)data)[base_index];
    }

  return (uint4)0;
}

/* for use inside filter functions
 * no channel mapping
 * no pointers to img metadata */
static float4
pocl_read_pixel_fast_f (int4 coord, int width, int height, int depth,
                        size_t row_pitch, size_t slice_pitch, int channel_type,
                        int order, void *data)
{
  size_t base_index
      = coord.x + (coord.y * row_pitch) + (coord.z * slice_pitch);

  if ((coord.x >= width || coord.x < 0)
      || ((height != 0) && (coord.y >= height || coord.y < 0))
      || ((depth != 0) && (coord.z >= depth || coord.z < 0)))
    {
      /* if out of bounds, return BORDER COLOR:
       * since pocl's basic/pthread device only
       * supports CL_A + CL_{RGBA combos},
       * the border color is always zeroes. */
      return (float4) (BORDER_COLOR_F);
    }

  if (order == CL_A)
    {
      float p = get_float_pixel (data, base_index, channel_type);
      return (float4) (0.0f, 0.0f, 0.0f, p);
    }

  return get_float4_pixel (data, base_index, channel_type);
}

/* for use inside filter functions
 * no channel mapping
 * no pointers to img metadata */
static int4
pocl_read_pixel_fast_i (int4 coord, int width, int height, int depth,
                        size_t row_pitch, size_t slice_pitch, int order,
                        int elem_size, void *data)
{
  int4 color;
  size_t base_index
      = coord.x + (coord.y * row_pitch) + (coord.z * slice_pitch);

  if ((coord.x >= width || coord.x < 0)
      || ((height != 0) && (coord.y >= height || coord.y < 0))
      || ((depth != 0) && (coord.z >= depth || coord.z < 0)))
    {
      /* if out of bounds, return BORDER COLOR:
       * since pocl's basic/pthread device only
       * supports CL_A + CL_{RGBA combos},
       * the border color is always zeroes. */
      return (int4) (BORDER_COLOR);
    }

  if (order == CL_A)
    {
      color = (int4)0;
      if (elem_size == 1)
        color.w = ((char *)data)[base_index];
      else if (elem_size == 2)
        color.w = ((short *)data)[base_index];
      else if (elem_size == 4)
        color.w = ((int *)data)[base_index];
      return color;
    }

  if (elem_size == 1)
    {
      return convert_int4 (((char4 *)data)[base_index]);
    }
  else if (elem_size == 2)
    {
      return convert_int4 (((short4 *)data)[base_index]);
    }
  else if (elem_size == 4)
    {
      return ((int4 *)data)[base_index];
    }
  return (int4)0;
}

/*************************************************************************/

static int4
get_image_array_offset (global dev_image_t *img, int4 uvw_after_rint,
                        int4 array_coord)
{
  int4 res = uvw_after_rint;
  if (img->_image_array_size > 0)
    {
      if (img->_height > 0)
        {
          res.z = clamp (array_coord.z, 0, (img->_image_array_size - 1));
          res.w = 0;
        }
      else
        {
          res.y = clamp (array_coord.y, 0, (img->_image_array_size - 1));
          res.z = 0;
          res.w = 0;
        }
    }
  return res;
}

/* array_coord must be unnormalized & repeats removed */
static int4
get_image_array_offset2 (global dev_image_t *img, int4 uvw_after_rint,
                         float4 array_coord)
{
  int4 res = uvw_after_rint;
  if (img->_image_array_size > 0)
    {
      if (img->_height > 0)
        {
          res.z = clamp (convert_int (floor (array_coord.z + 0.5f)), 0,
                         (img->_image_array_size - 1));
          res.w = 0;
        }
      else
        {
          res.y = clamp (convert_int (floor (array_coord.y + 0.5f)), 0,
                         (img->_image_array_size - 1));
          res.z = 0;
          res.w = 0;
        }
    }
  return res;
}

/* RET: (int4) (img.x{,y,z}, array_size, 0 {,0 ...} ) */
static int4
get_image_array_size (global dev_image_t *img)
{
  int4 imgsize = (int4) (img->_width, img->_height, img->_depth, 0);
  if (img->_image_array_size > 0)
    {
      if (img->_height > 0)
        imgsize.z = img->_image_array_size;
      else
        imgsize.y = img->_image_array_size;
    }
  return imgsize;
}
/*************************************************************************/

/* full read with channel map conversion etc  */
/* Reads a four element pixel from image pointed by integer coords.
 * Returns Border color (0) for out-of-range reads. This is OK since
 * reads behind border should either return border color, or are undefined */
static uint4
pocl_read_pixel (global dev_image_t *img, int4 coord)
{
  uint4 color;
  int width = img->_width;
  int height = img->_height;
  int depth = img->_depth;
  int num_channels = img->_num_channels;
  int order = img->_order;
  int elem_size = img->_elem_size;
  int channel_type = img->_data_type;
  void *data = img->_data;
  size_t elem_bytes = num_channels * elem_size;
  size_t row_pitch = img->_row_pitch / elem_bytes;
  size_t slice_pitch = img->_slice_pitch / elem_bytes;

  if ((channel_type == CL_SIGNED_INT8) || (channel_type == CL_SIGNED_INT16)
      || (channel_type == CL_SIGNED_INT32))
    color = as_uint4 (pocl_read_pixel_fast_i (coord, width, height, depth,
                                              row_pitch, slice_pitch, order,
                                              elem_size, data));
  else if ((channel_type == CL_UNSIGNED_INT8)
           || (channel_type == CL_UNSIGNED_INT16)
           || (channel_type == CL_UNSIGNED_INT32))
    color = pocl_read_pixel_fast_ui (coord, width, height, depth, row_pitch,
                                     slice_pitch, order, elem_size, data);
  else // TODO unsupported channel types
    color = as_uint4 (pocl_read_pixel_fast_f (coord, width, height, depth,
                                              row_pitch, slice_pitch,
                                              channel_type, order, data));

  return map_channels (color, order);
}

/* Transforms coords based on image addressing mode */
static int4
pocl_address_mode (global dev_image_t *img, int4 input_coord,
                   dev_sampler_t samp)
{
  if (samp & CLK_ADDRESS_CLAMP_TO_EDGE)
    {
      int4 max_clamp = max (
          (int4) (img->_width - 1, img->_height - 1, img->_depth - 1, 0),
          (int4)0);
      return clamp (input_coord, (int4) (0), max_clamp);
    }

  if (samp & CLK_ADDRESS_CLAMP)
    {
      int4 max_clamp
          = max ((int4) (img->_width, img->_height, img->_depth, 0), (int4)0);
      return clamp (input_coord, (int4) (-1), max_clamp);
    }

  return input_coord;
}

/*************************************************************************/

static float4
read_pixel_linear_3d_float (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                            int width, int height, int depth, int channel_type,
                            size_t row_pitch, size_t slice_pitch, int order,
                            void *data)
{
  // 3D image
  // T = (1 – a) * (1 – b) * (1 – c) * Ti0j0k0
  return (
      one_m.x * one_m.y * one_m.z
          * pocl_read_pixel_fast_f (ijk0, width, height, depth, row_pitch,
                                    slice_pitch, channel_type, order, data)
      // + a * (1 – b) * (1 – c) * Ti1j0k0
      + abc.x * one_m.y * one_m.z
            * pocl_read_pixel_fast_f ((int4) (ijk1.x, ijk0.y, ijk0.z, 0),
                                      width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data)
      // + (1 – a) * b * (1 – c) * Ti0j1k0
      + one_m.x * abc.y * one_m.z
            * pocl_read_pixel_fast_f ((int4) (ijk0.x, ijk1.y, ijk0.z, 0),
                                      width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data)
      // + a * b * (1 – c) * Ti1j1k0
      + abc.x * abc.y * one_m.z
            * pocl_read_pixel_fast_f ((int4) (ijk1.x, ijk1.y, ijk0.z, 0),
                                      width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data)
      // + (1 – a) * (1 – b) * c * Ti0j0k1
      + one_m.x * one_m.y * abc.z
            * pocl_read_pixel_fast_f ((int4) (ijk0.x, ijk0.y, ijk1.z, 0),
                                      width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data)
      // + a * (1 – b) * c * Ti1j0k1
      + abc.x * one_m.y * abc.z
            * pocl_read_pixel_fast_f ((int4) (ijk1.x, ijk0.y, ijk1.z, 0),
                                      width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data)
      // + (1 – a) * b * c * Ti0j1k1
      + one_m.x * abc.y * abc.z
            * pocl_read_pixel_fast_f ((int4) (ijk0.x, ijk1.y, ijk1.z, 0),
                                      width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data)
      // + a * b * c * Ti1j1k1
      + abc.x * abc.y * abc.z
            * pocl_read_pixel_fast_f (ijk1, width, height, depth, row_pitch,
                                      slice_pitch, channel_type, order, data));
}

/* TODO: float * convert_flaot(UINT32) is imprecise, so reading from images
 * with 32bit channel types may return quite bad results.
 */

static uint4
read_pixel_linear_3d_uint (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                           int width, int height, int depth, size_t row_pitch,
                           size_t slice_pitch, int order, int elem_size,
                           void *data)
{
  // 3D image
  // T = (1 – a) * (1 – b) * (1 – c) * Ti0j0k0
  return convert_uint4 (
      one_m.x * one_m.y * one_m.z * convert_float4 (pocl_read_pixel_fast_ui (
                                        ijk0, width, height, depth, row_pitch,
                                        slice_pitch, order, elem_size, data))
      // + a * (1 – b) * (1 – c) * Ti1j0k0
      + abc.x * one_m.y * one_m.z
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk1.x, ijk0.y, ijk0.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * b * (1 – c) * Ti0j1k0
      + one_m.x * abc.y * one_m.z
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk0.x, ijk1.y, ijk0.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * b * (1 – c) * Ti1j1k0
      + abc.x * abc.y * one_m.z
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk1.x, ijk1.y, ijk0.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * (1 – b) * c * Ti0j0k1
      + one_m.x * one_m.y * abc.z
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk0.x, ijk0.y, ijk1.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * (1 – b) * c * Ti1j0k1
      + abc.x * one_m.y * abc.z
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk1.x, ijk0.y, ijk1.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * b * c * Ti0j1k1
      + one_m.x * abc.y * abc.z
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk0.x, ijk1.y, ijk1.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * b * c * Ti1j1k1
      + abc.x * abc.y * abc.z * convert_float4 (pocl_read_pixel_fast_ui (
                                    ijk1, width, height, depth, row_pitch,
                                    slice_pitch, order, elem_size, data)));
}

static int4
read_pixel_linear_3d_int (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                          int width, int height, int depth, size_t row_pitch,
                          size_t slice_pitch, int order, int elem_size,
                          void *data)
{
  // 3D image
  // T = (1 – a) * (1 – b) * (1 – c) * Ti0j0k0
  return convert_int4 (
      one_m.x * one_m.y * one_m.z * convert_float4 (pocl_read_pixel_fast_i (
                                        ijk0, width, height, depth, row_pitch,
                                        slice_pitch, order, elem_size, data))
      // + a * (1 – b) * (1 – c) * Ti1j0k0
      + abc.x * one_m.y * one_m.z
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk1.x, ijk0.y, ijk0.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * b * (1 – c) * Ti0j1k0
      + one_m.x * abc.y * one_m.z
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk0.x, ijk1.y, ijk0.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * b * (1 – c) * Ti1j1k0
      + abc.x * abc.y * one_m.z
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk1.x, ijk1.y, ijk0.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * (1 – b) * c * Ti0j0k1
      + one_m.x * one_m.y * abc.z
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk0.x, ijk0.y, ijk1.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * (1 – b) * c * Ti1j0k1
      + abc.x * one_m.y * abc.z
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk1.x, ijk0.y, ijk1.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * b * c * Ti0j1k1
      + one_m.x * abc.y * abc.z
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk0.x, ijk1.y, ijk1.z, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * b * c * Ti1j1k1
      + abc.x * abc.y * abc.z * convert_float4 (pocl_read_pixel_fast_i (
                                    ijk1, width, height, depth, row_pitch,
                                    slice_pitch, order, elem_size, data)));
}

static uint4
read_pixel_linear_3d (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                      int width, int height, int depth, int channel_type,
                      size_t row_pitch, size_t slice_pitch, int order,
                      int elem_size, void *data)
{
  // TODO unsupported channel types
  if ((channel_type == CL_SIGNED_INT8) || (channel_type == CL_SIGNED_INT16)
      || (channel_type == CL_SIGNED_INT32))
    return as_uint4 (read_pixel_linear_3d_int (
        abc, one_m, ijk0, ijk1, width, height, depth, row_pitch, slice_pitch,
        order, elem_size, data));
  if ((channel_type == CL_UNSIGNED_INT8) || (channel_type == CL_UNSIGNED_INT16)
      || (channel_type == CL_UNSIGNED_INT32))
    return read_pixel_linear_3d_uint (abc, one_m, ijk0, ijk1, width, height,
                                      depth, row_pitch, slice_pitch, order,
                                      elem_size, data);
  return as_uint4 (read_pixel_linear_3d_float (
      abc, one_m, ijk0, ijk1, width, height, depth, channel_type, row_pitch,
      slice_pitch, order, data));
}

/*************************************************************************/

static float4
read_pixel_linear_2d_float (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                            int width, int height, int depth, int channel_type,
                            size_t row_pitch, size_t slice_pitch, int order,
                            void *data)
{
  // 2D image
  // T = (1 – a) * (1 – b) * Ti0j0
  return (
      one_m.x * one_m.y * pocl_read_pixel_fast_f (ijk0, width, height, depth,
                                                  row_pitch, slice_pitch,
                                                  channel_type, order, data)
      // + a * (1 – b) * Ti1j0
      + abc.x * one_m.y
            * pocl_read_pixel_fast_f ((int4) (ijk1.x, ijk0.y, 0, 0), width,
                                      height, depth, row_pitch, slice_pitch,
                                      channel_type, order, data)
      // + (1 – a) * b * Ti0j1
      + one_m.x * abc.y
            * pocl_read_pixel_fast_f ((int4) (ijk0.x, ijk1.y, 0, 0), width,
                                      height, depth, row_pitch, slice_pitch,
                                      channel_type, order, data)
      // + a * b * Ti1j1
      + abc.x * abc.y * pocl_read_pixel_fast_f (ijk1, width, height, depth,
                                                row_pitch, slice_pitch,
                                                channel_type, order, data));
}

/* TODO: float * convert_flaot(UINT32) is imprecise, so reading from images
 * with 32bit channel types may return quite bad results.
 */

static uint4
read_pixel_linear_2d_uint (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                           int width, int height, int depth, size_t row_pitch,
                           size_t slice_pitch, int order, int elem_size,
                           void *data)
{
  // 2D image
  // T = (1 – a) * (1 – b) * Ti0j0
  return convert_uint4 (
      one_m.x * one_m.y * convert_float4 (pocl_read_pixel_fast_ui (
                              ijk0, width, height, depth, row_pitch,
                              slice_pitch, order, elem_size, data))
      // + a * (1 – b) * Ti1j0
      + abc.x * one_m.y
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk1.x, ijk0.y, 0, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * b * Ti0j1
      + one_m.x * abc.y
            * convert_float4 (pocl_read_pixel_fast_ui (
                  (int4) (ijk0.x, ijk1.y, 0, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * b * Ti1j1
      + abc.x * abc.y * convert_float4 (pocl_read_pixel_fast_ui (
                            ijk1, width, height, depth, row_pitch, slice_pitch,
                            order, elem_size, data)));
}

static int4
read_pixel_linear_2d_int (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                          int width, int height, int depth, size_t row_pitch,
                          size_t slice_pitch, int order, int elem_size,
                          void *data)
{
  // 2D image
  // T = (1 – a) * (1 – b) * Ti0j0
  return convert_int4 (
      one_m.x * one_m.y * convert_float4 (pocl_read_pixel_fast_i (
                              ijk0, width, height, depth, row_pitch,
                              slice_pitch, order, elem_size, data))
      // + a * (1 – b) * Ti1j0
      + abc.x * one_m.y
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk1.x, ijk0.y, 0, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + (1 – a) * b * Ti0j1
      + one_m.x * abc.y
            * convert_float4 (pocl_read_pixel_fast_i (
                  (int4) (ijk0.x, ijk1.y, 0, 0), width, height, depth,
                  row_pitch, slice_pitch, order, elem_size, data))
      // + a * b * Ti1j1
      + abc.x * abc.y * convert_float4 (pocl_read_pixel_fast_i (
                            ijk1, width, height, depth, row_pitch, slice_pitch,
                            order, elem_size, data)));
}

static uint4
read_pixel_linear_2d (float4 abc, float4 one_m, int4 ijk0, int4 ijk1,
                      int width, int height, int depth, int channel_type,
                      size_t row_pitch, size_t slice_pitch, int order,
                      int elem_size, void *data)
{
  // TODO unsupported channel types
  if ((channel_type == CL_SIGNED_INT8) || (channel_type == CL_SIGNED_INT16)
      || (channel_type == CL_SIGNED_INT32))
    return as_uint4 (read_pixel_linear_2d_int (
        abc, one_m, ijk0, ijk1, width, height, depth, row_pitch, slice_pitch,
        order, elem_size, data));
  if ((channel_type == CL_UNSIGNED_INT8) || (channel_type == CL_UNSIGNED_INT16)
      || (channel_type == CL_UNSIGNED_INT32))
    return read_pixel_linear_2d_uint (abc, one_m, ijk0, ijk1, width, height,
                                      depth, row_pitch, slice_pitch, order,
                                      elem_size, data);
  return as_uint4 (read_pixel_linear_2d_float (
      abc, one_m, ijk0, ijk1, width, height, depth, channel_type, row_pitch,
      slice_pitch, order, data));
}

/*************************************************************************/

/* These magic constant should be converted to some sort of
 * error signaling */
#define INVALID_SAMPLER_ADDRMODE (uint4) (0x1111)
#define INVALID_SAMPLER_FILTER (uint4) (0x2222)
#define INVALID_SAMPLER_NORMAL (uint4) (0x3333)

static uint4
nonrepeat_filter (global dev_image_t *img, float4 orig_coord,
                  dev_sampler_t samp)
{
  float4 coord = orig_coord;
  if (samp & CLK_NORMALIZED_COORDS_TRUE)
    {
      float4 imgsize = (float4) (img->_width, img->_height, img->_depth, 0);
      coord *= imgsize;
    }

  int num_channels = img->_num_channels;
  int elem_size = img->_elem_size;
  size_t elem_bytes = num_channels * elem_size;
  size_t row_pitch = img->_row_pitch / elem_bytes;
  size_t slice_pitch = img->_slice_pitch / elem_bytes;

  if (samp & CLK_FILTER_NEAREST)
    {
      int4 final_coord
          = pocl_address_mode (img, convert_int4 (floor (coord)), samp);
      int4 array_coord = get_image_array_offset (img, final_coord,
                                                 convert_int4 (rint (coord)));
      return pocl_read_pixel (img, array_coord);
    }
  else if (samp & CLK_FILTER_LINEAR)
    {
      float4 r0 = floor (coord - (float4) (0.5f)); // ijk0, address mod
      float4 r1 = floor (coord - (float4) (0.5f))
                  + (float4) (1.0f); // ijk1, address mod
      int4 ijk0 = pocl_address_mode (img, convert_int4 (r0), samp);
      int4 ijk1 = pocl_address_mode (img, convert_int4 (r1), samp);
      float4 unused;
      float4 abc = fract ((coord - (float4) (0.5f)), &unused);
      float4 one_m = (float4) (1.0f) - abc;
      uint4 res;
      if (img->_depth != 0)
        {
          res = read_pixel_linear_3d (
              abc, one_m, ijk0, ijk1, img->_width, img->_height, img->_depth,
              img->_data_type, row_pitch, slice_pitch, img->_order,
              img->_elem_size, img->_data);
        }
      else
        {
          res = read_pixel_linear_2d (
              abc, one_m, ijk0, ijk1, img->_width, img->_height, img->_depth,
              img->_data_type, row_pitch, slice_pitch, img->_order,
              img->_elem_size, img->_data);
        }
      return map_channels (res, img->_order);
    }
  else
    {
      // this should never happen - filter can only be LINEAR/NEAREST
      return INVALID_SAMPLER_FILTER;
    }
}

static uint4
repeat_filter (global dev_image_t *img, float4 coord, dev_sampler_t samp)
{
  int num_channels = img->_num_channels;
  int elem_size = img->_elem_size;
  size_t elem_bytes = num_channels * elem_size;
  size_t row_pitch = img->_row_pitch / elem_bytes;
  size_t slice_pitch = img->_slice_pitch / elem_bytes;

  if (samp & CLK_FILTER_NEAREST)
    {
      /*
         uvw = (str – floor(str)) * whd
         ijk = (int)floor(uvw)
         if (ijk > whd – 1)
           ijk = ijk – whd
         ... same for 3 coords
      */
      int4 maxcoord = (int4) (img->_width, img->_height, img->_depth, 0);
      float4 whd = convert_float4 (maxcoord);
      float4 uvw = (coord - floor (coord)) * whd;
      int4 ijk = convert_int4 (floor (uvw));
      int4 final_coord = select (ijk, (ijk - maxcoord), (ijk >= maxcoord));
      int4 array_coord = get_image_array_offset (img, final_coord, ijk);
      return pocl_read_pixel (img, array_coord);
    }
  else if (samp & CLK_FILTER_LINEAR)
    {
      /*
          u = (s – floor(s)) * wt
          i0 = (int)floor(u – 0.5)
          i1 = i0 + 1
          if (i0 < 0)
           i0 = wt + i0
          if (i1 > wt – 1)
           i1 = i1 – wt
      */
      int4 maxcoord = (int4) (img->_width, img->_height, img->_depth, 1);
      float4 whd = convert_float4 (maxcoord);
      float4 uvw = (coord - floor (coord)) * whd;
      int4 ijk0 = convert_int4 (floor (uvw - (float4) (0.5f)));
      int4 ijk1 = ijk0 + (int4) (1);
      ijk0 = select (ijk0, (ijk0 + maxcoord), (ijk0 < (int4) (0)));
      maxcoord = max (maxcoord, (int4)1);
      ijk1 = ijk1 % maxcoord;
      float4 unused;
      float4 abc = fract ((uvw - (float4) (0.5f)), &unused);
      float4 one_m = (float4) (1.0f) - abc;
      uint4 res;
      if (img->_depth != 0)
        {
          res = read_pixel_linear_3d (
              abc, one_m, ijk0, ijk1, img->_width, img->_height, img->_depth,
              img->_data_type, row_pitch, slice_pitch, img->_order,
              img->_elem_size, img->_data);
        }
      else
        {
          res = read_pixel_linear_2d (
              abc, one_m, ijk0, ijk1, img->_width, img->_height, img->_depth,
              img->_data_type, row_pitch, slice_pitch, img->_order,
              img->_elem_size, img->_data);
        }
      return map_channels (res, img->_order);
    }
  else
    {
      // this should never happen - filter can only be LINEAR/NEAREST
      return INVALID_SAMPLER_FILTER;
    }
}

static uint4
mirrored_repeat_filter (global dev_image_t *img, float4 coord,
                        dev_sampler_t samp)
{
  int num_channels = img->_num_channels;
  int elem_size = img->_elem_size;
  size_t elem_bytes = num_channels * elem_size;
  size_t row_pitch = img->_row_pitch / elem_bytes;
  size_t slice_pitch = img->_slice_pitch / elem_bytes;

  if (samp & CLK_FILTER_NEAREST)
    {
      /*
        s’ = 2.0f * rint(0.5f * s)
        s’ = fabs(s – s’)
        u = s’ * wt
        i = (int)floor(u)
        i = min(i, wt – 1)
      */
      float4 ss = (float4) (2.0f) * rint ((float4) (0.5f) * coord);
      ss = fabs (coord - ss);
      int4 maxcoord = (int4) (img->_width, img->_height, img->_depth, 1);
      float4 uvw = ss * convert_float4 (maxcoord);
      int4 ijk = convert_int4 (floor (uvw));
      int4 final_coord = min (ijk, (maxcoord - (int4) (1)));
      int4 array_coord = get_image_array_offset (img, final_coord, ijk);
      return pocl_read_pixel (img, array_coord);
    }
  else if (samp & CLK_FILTER_LINEAR)
    {
      /*
        s’ = 2.0f * rint(0.5f * s)
        s’ = fabs(s – s’)
        u = s’ * wt
        i0 = (int)floor(u – 0.5f)
        i1 = i0 + 1
        i0 = max(i0, 0)
        i1 = min(i1, wt – 1)
      */
      float4 ss = (float4) (2.0f) * rint ((float4) (0.5f) * coord);
      ss = fabs (coord - ss);
      int4 maxcoord = (int4) (img->_width, img->_height, img->_depth, 1);
      float4 uvw = ss * convert_float4 (maxcoord);
      int4 ijk0 = convert_int4 (floor (uvw - (float4) (0.5f)));
      int4 ijk1 = ijk0 + (int4) (1);
      ijk0 = max (ijk0, (int4)0);
      ijk1 = min (ijk1, (maxcoord - (int4) (1)));
      float4 unused;
      float4 abc = fract ((uvw - (float4) (0.5f)), &unused);
      float4 one_m = (float4) (1.0f) - abc;
      uint4 res;
      if (img->_depth != 0)
        {
          res = read_pixel_linear_3d (
              abc, one_m, ijk0, ijk1, img->_width, img->_height, img->_depth,
              img->_data_type, row_pitch, slice_pitch, img->_order,
              img->_elem_size, img->_data);
        }
      else
        {
          res = read_pixel_linear_2d (
              abc, one_m, ijk0, ijk1, img->_width, img->_height, img->_depth,
              img->_data_type, row_pitch, slice_pitch, img->_order,
              img->_elem_size, img->_data);
        }
      return map_channels (res, img->_order);
    }
  else
    {
      // this should never happen - filter can only be LINEAR/NEAREST
      return INVALID_SAMPLER_FILTER;
    }
}

/*************************************************************************/
/* read pixel with float coordinates */
static uint4
pocl_read_pixel_floatc (global dev_image_t *img, float4 coord,
                        dev_sampler_t samp)
{
  if (samp & CLK_ADDRESS_REPEAT)
    return repeat_filter (img, coord, samp);
  else if (samp & CLK_ADDRESS_MIRRORED_REPEAT)
    return mirrored_repeat_filter (img, coord, samp);
  else
    return nonrepeat_filter (img, coord, samp);
}

/*************************************************************************/
/* read pixel with int coordinates
 * from Spec:
 *
 * Furthermore, the read_imagei and read_imageui calls that take integer
 * coordinates must use a sampler with normalized coordinates set to
 * CLK_NORMALIZED_COORDS_FALSE and addressing mode set to
 * CLK_ADDRESS_CLAMP_TO_EDGE, CLK_ADDRESS_CLAMP or CLK_ADDRESS_NONE;
 * otherwise the values returned are undefined.
*/

static uint4
pocl_read_pixel_intc (global dev_image_t *img, int4 coord, dev_sampler_t samp)
{
  if (samp & CLK_NORMALIZED_COORDS_TRUE)
    return INVALID_SAMPLER_NORMAL;
  if ((samp & CLK_ADDRESS_REPEAT) || (samp & CLK_ADDRESS_MIRRORED_REPEAT))
    return INVALID_SAMPLER_ADDRMODE;

  int4 final_coord = pocl_address_mode (img, coord, samp);
  int4 array_coord = get_image_array_offset (img, final_coord, coord);
  return pocl_read_pixel (img, array_coord);
}

/******************* DONE *************************************************/
/* read pixel with float coordinates, WITHOUT sampler
 * from Spec:
 * The samplerless read image functions behave exactly as the corresponding
 * read image functions that take integer coordinates and a sampler with
 * filter mode set to CLK_FILTER_NEAREST, normalized coordinates set to
 * CLK_NORMALIZED_COORDS_FALSE and addressing mode to CLK_ADDRESS_NONE.
 */

static uint4
pocl_read_pixel_intc_samplerless (global dev_image_t *img, int4 coord)
{
  dev_sampler_t samp
      = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE;

  int4 final_coord = pocl_address_mode (img, coord, samp);
  int4 array_coord = get_image_array_offset (img, final_coord, coord);
  return pocl_read_pixel (img, array_coord);
}

/*************************************************************************/

#if __clang_major__ > 3
/* After Clang 4.0, the sampler_t is passed as an opaque struct (ptr)
 which we convert to int32 with the LLVM pass HandleSamplerInitialization. */
#define READ_SAMPLER                                                    \
    dev_sampler_t s = *__builtin_astype(sampler, dev_sampler_t*);
#else
/* Before Clang 4.0, the sampler_t was passed as an int32. */
#define READ_SAMPLER                                                    \
    dev_sampler_t s = __builtin_astype(sampler, dev_sampler_t);
#endif

/* Implementation for read_image with any image data type and int coordinates
   __IMGTYPE__ = image type (image2d_t, ...)
   __RETVAL__  = return value (int4 or uint4 float4)
   __POSTFIX__ = function name postfix (i, ui, f)
   __COORD__   = coordinate type (int, int2, int4)
*/

#define IMPLEMENT_READ_INT4_IMAGE_INT_COORD(__IMGTYPE__, __RETVAL__,          \
                                            __POSTFIX__, __COORD__)           \
  __RETVAL__ _CL_OVERLOADABLE read_image##__POSTFIX__ (                       \
      __IMGTYPE__ image, sampler_t sampler, __COORD__ coord)                  \
  {                                                                           \
    int4 coord4;                                                              \
    INITCOORD##__COORD__ (coord4, coord);                                     \
    global dev_image_t *i_ptr                                                 \
        = __builtin_astype (image, global dev_image_t *);                     \
    READ_SAMPLER                                                              \
    uint4 color = pocl_read_pixel_intc (i_ptr, coord4, s);                    \
    return as_##__RETVAL__ (color);                                           \
  }

#define IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD(__IMGTYPE__, __COORD__)         \
  float4 _CL_OVERLOADABLE read_imagef (__IMGTYPE__ image, sampler_t sampler,  \
                                       __COORD__ coord)                       \
  {                                                                           \
    int4 coord4;                                                              \
    INITCOORD##__COORD__ (coord4, coord);                                     \
    global dev_image_t *i_ptr                                                 \
        = __builtin_astype (image, global dev_image_t *);                     \
    READ_SAMPLER                                                              \
    uint4 color = pocl_read_pixel_intc (i_ptr, coord4, s);                    \
    return as_float4 (color);                                                 \
  }

#define IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD(__IMGTYPE__, __COORD__)       \
  float4 _CL_OVERLOADABLE read_imagef (__IMGTYPE__ image, sampler_t sampler,  \
                                       __COORD__ coord)                       \
  {                                                                           \
    float4 coord4;                                                            \
    INITCOORD##__COORD__ (coord4, coord);                                     \
    global dev_image_t *i_ptr                                                 \
        = __builtin_astype (image, global dev_image_t *);                     \
    READ_SAMPLER                                                              \
    uint4 color = pocl_read_pixel_floatc (i_ptr, coord4, s);                  \
    return as_float4 (color);                                                 \
  }

#define IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD(__IMGTYPE__, __RETVAL__,        \
                                              __POSTFIX__, __COORD__)         \
  __RETVAL__ _CL_OVERLOADABLE read_image##__POSTFIX__ (                       \
      __IMGTYPE__ image, sampler_t sampler, __COORD__ coord)                  \
  {                                                                           \
    float4 coord4;                                                            \
    INITCOORD##__COORD__ (coord4, coord);                                     \
    global dev_image_t *i_ptr                                                 \
        = __builtin_astype (image, global dev_image_t *);                     \
    READ_SAMPLER                                                              \
    uint4 color = pocl_read_pixel_floatc (i_ptr, coord4, s);                  \
    return as_##__RETVAL__ (color);                                           \
  }

/* NO Sampler Implementation for read_image with any image data type
   and int coordinates
   __IMGTYPE__ = image type (image2d_t, ...)
   __RETVAL__  = return value (int4 or uint4 float4)
   __POSTFIX__ = function name postfix (i, ui, f)
   __COORD__   = coordinate type (int, int2, int4)
   OCL 1.2 Spec Says
   The samplerless read image functions behave exactly as the
   corresponding read image functions that take integer
   coordinates and a sampler with filter mode set to
   CLK_FILTER_NEAREST, normalized coordinates set to
   CLK_NORMALIZED_COORDS_FALSE and
   addressing mode to CLK_ADDRESS_NONE.
   CLK_ADDRESS_NONE – for this addressing mode the programmer guarantees that
   the image coordinates used to sample elements of the image refer to a
   location inside the image; otherwise the results are undefined.
   Thus we do not need out of bound check for now.
   If we need sampler for other cases it has to be defined as default sampler:
   const sampler_t defualt_sampler = CLK_NORMALIZED_COORDS_FALSE |
                                  CLK_ADDRESS_NONE |
                                  CLK_FILTER_NEAREST;
*/

#define IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER(                        \
    __IMGTYPE__, __RETVAL__, __POSTFIX__, __COORD__)                          \
  __RETVAL__ _CL_OVERLOADABLE read_image##__POSTFIX__ (__IMGTYPE__ image,     \
                                                       __COORD__ coord)       \
  {                                                                           \
    int4 coord4;                                                              \
    INITCOORD##__COORD__ (coord4, coord);                                     \
    global dev_image_t *i_ptr                                                 \
        = __builtin_astype (image, global dev_image_t *);                     \
    uint4 color = pocl_read_pixel_intc_samplerless (i_ptr, coord4);           \
    return as_##__RETVAL__ (color);                                           \
  }

#define IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER(__IMGTYPE__,          \
                                                        __COORD__)            \
  float4 _CL_OVERLOADABLE read_imagef (__IMGTYPE__ image, __COORD__ coord)    \
  {                                                                           \
    int4 coord4;                                                              \
    INITCOORD##__COORD__ (coord4, coord);                                     \
    global dev_image_t *i_ptr                                                 \
        = __builtin_astype (image, global dev_image_t *);                     \
    uint4 color = pocl_read_pixel_intc_samplerless (i_ptr, coord4);           \
    return as_float4 (color);                                                 \
  }

/* NO sampler */

IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_t, int)
/* IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_buffer_t, int) */
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_array_t, int2)

IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image2d_t, int2)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image2d_array_t,
                                                 int4)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image3d_t, int4)



IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_t, uint4, ui,
                                               int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_t, int4, i,
                                               int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_array_t, uint4, ui,
                                               int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_array_t, int4, i,
                                               int2)
/*IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_buffer_t, uint4, ui,
 *                                               int)
 *IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image1d_buffer_t, int4, i,
 *                                               int)
 */

IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image2d_t, uint4, ui,
                                               int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image2d_t, int4, i,
                                               int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image2d_array_t,
                                               uint4, ui, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image2d_array_t, int4,
                                               i, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image3d_t, uint4, ui,
                                               int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RO_AQ image3d_t, int4, i,
                                               int4)

/* float4 img + float coords + sampler */

IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image1d_t, float)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image1d_array_t, float2)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image2d_t, float2)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image2d_array_t, float4)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image3d_t, float4)

/* float4 img + int coords + sampler */

IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RO_AQ image1d_t, int)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RO_AQ image1d_array_t, int2)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RO_AQ image2d_t, int2)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RO_AQ image2d_array_t, int4)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RO_AQ image3d_t, int4)

/* int4 img + float coords + sampler */

IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image1d_t, uint4, ui, float)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image1d_t, int4, i, float)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image1d_array_t, uint4, ui,
                                       float2)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image1d_array_t, int4, i,
                                       float2)

IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image2d_t, uint4, ui, float2)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image2d_t, int4, i, float2)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image2d_array_t, uint4, ui,
                                       float4)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image2d_array_t, int4, i,
                                       float4)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image3d_t, uint4, ui, float4)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RO_AQ image3d_t, int4, i, float4)

/* int4 img + int coords + sampler */

IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image1d_t, uint4, ui, int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image1d_t, int4, i, int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image1d_array_t, uint4, ui,
                                     int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image1d_array_t, int4, i, int2)

IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image2d_t, uint4, ui, int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image2d_t, int4, i, int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image2d_array_t, uint4, ui,
                                     int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image2d_array_t, int4, i, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image3d_t, uint4, ui, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RO_AQ image3d_t, int4, i, int4)

/******************************************************************************/
/******************************************************************************/

#ifdef CLANG_HAS_RW_IMAGES

/* NO sampler */

IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_t, int)
/* IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_buffer_t, int) */
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_array_t, int2)

IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image2d_t, int2)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image2d_array_t,
                                                 int4)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image3d_t, int4)


IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_t, uint4, ui,
                                               int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_t, int4, i,
                                               int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_array_t, uint4, ui,
                                               int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_array_t, int4, i,
                                               int2)
/*IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_buffer_t, uint4, ui,
 *                                               int)
 *IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image1d_buffer_t, int4, i,
 *                                               int)
 */


IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image2d_t, uint4, ui,
                                               int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image2d_t, int4, i,
                                               int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image2d_array_t,
                                               uint4, ui, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image2d_array_t, int4,
                                               i, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image3d_t, uint4, ui,
                                               int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD_NOSAMPLER (IMG_RW_AQ image3d_t, int4, i,
                                               int4)

/* float4 img + float coords + sampler */

IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image1d_t, float)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image1d_array_t, float2)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image2d_t, float2)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image2d_array_t, float4)
IMPLEMENT_READ_FLOAT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image3d_t, float4)

/* float4 img + int coords + sampler */

IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RW_AQ image1d_t, int)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RW_AQ image1d_array_t, int2)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RW_AQ image2d_t, int2)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RW_AQ image2d_array_t, int4)
IMPLEMENT_READ_FLOAT4_IMAGE_INT_COORD (IMG_RW_AQ image3d_t, int4)

/* int4 img + float coords + sampler */

IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image1d_t, uint4, ui, float)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image1d_t, int4, i, float)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image1d_array_t, uint4, ui,
                                       float2)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image1d_array_t, int4, i,
                                       float2)

IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image2d_t, uint4, ui, float2)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image2d_t, int4, i, float2)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image2d_array_t, uint4, ui,
                                       float4)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image2d_array_t, int4, i,
                                       float4)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image3d_t, uint4, ui, float4)
IMPLEMENT_READ_INT4_IMAGE_FLOAT_COORD (IMG_RW_AQ image3d_t, int4, i, float4)

/* int4 img + int coords + sampler */

IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image1d_t, uint4, ui, int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image1d_t, int4, i, int)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image1d_array_t, uint4, ui,
                                     int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image1d_array_t, int4, i, int2)

IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image2d_t, uint4, ui, int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image2d_t, int4, i, int2)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image2d_array_t, uint4, ui,
                                     int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image2d_array_t, int4, i, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image3d_t, uint4, ui, int4)
IMPLEMENT_READ_INT4_IMAGE_INT_COORD (IMG_RW_AQ image3d_t, int4, i, int4)

#endif
