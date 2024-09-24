/* pocl_dbk_khr_jpeg_cpu.c - JPEG Defined Built-in Kernel functions for CPU
   devices.

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

#include "pocl_dbk_khr_jpeg_cpu.h"
#include "common_utils.h"

#ifdef HAVE_LIBJPEG_TURBO

#include <turbojpeg.h>

typedef struct
{
  tjhandle tj_handle;
  unsigned char *tj_buffer;
  /* This value is used to store the buffer size when the compression function
   * reallocates it. */
  size_t jt_buffer_size;
} jpeg_encode_state_t;

typedef struct
{
  tjhandle tj_handle;
} jpeg_decode_state_t;

int
pocl_cpu_execute_dbk_khr_jpeg_encode (cl_program program,
                                      cl_kernel kernel,
                                      pocl_kernel_metadata_t *meta,
                                      cl_uint dev_i,
                                      struct pocl_argument *arguments)
{
  cl_device_id dev = program->devices[dev_i];
  cl_dbk_attributes_exp_jpeg_encode *attributes
    = (cl_dbk_attributes_exp_jpeg_encode *)meta->builtin_kernel_attrs;
  unsigned mem_id = dev->global_mem_id;
  void *image = pocl_cpu_get_ptr (&arguments[0], mem_id);
  void *jpeg = pocl_cpu_get_ptr (&arguments[1], mem_id);
  uint64_t *jpeg_size = pocl_cpu_get_ptr (&arguments[2], mem_id);

  jpeg_encode_state_t *state = kernel->data[dev_i];

  int status;
  status = tj3Compress8 (state->tj_handle, image, attributes->width, 0,
                         attributes->height, TJPF_RGB, &(state->tj_buffer),
                         &(state->jt_buffer_size));
  if (status != 0)
    {
      POCL_MSG_ERR ("tj3Compress8 failed: %s (fatal? %d).\n",
                    tj3GetErrorStr (state->tj_handle),
                    tj3GetErrorCode (state->tj_handle));
      return CL_OUT_OF_RESOURCES;
    }
  /* TODO: remove this once the handle is configured with JPEG no realloc. */
  cl_mem jpeg_mem = *(cl_mem *)(arguments[0].value);
  if (jpeg_mem->size < state->jt_buffer_size)
    {
      POCL_MSG_ERR ("pocl_cpu_execut_dbk_khr_jpeg_encode, "
                    "compressed JPEG image is larger than destination.\n ");
      assert (0 && "improper parameters");
      return CL_OUT_OF_RESOURCES;
    }
  memcpy (jpeg, state->tj_buffer, state->jt_buffer_size);
  *jpeg_size = state->jt_buffer_size;
  return status;
}

void *
pocl_cpu_init_dbk_khr_jpeg_encode (void const *attributes, int *status)
{

  jpeg_encode_state_t *state
    = (jpeg_encode_state_t *)calloc (1, sizeof (jpeg_encode_state_t));
  state->tj_handle = tj3Init (TJINIT_COMPRESS);
  if (state->tj_handle == NULL)
    {
      const char *err_str = tj3GetErrorStr (state->tj_handle);
      POCL_MSG_ERR ("Create compress tj_handle failed: %s (fatal? %d).\n",
                    err_str, tj3GetErrorCode (state->tj_handle));
      *status = CL_OUT_OF_RESOURCES;
      return NULL;
    }

  int jpeg_status;
  cl_dbk_attributes_exp_jpeg_encode *jpeg_attr
    = (cl_dbk_attributes_exp_jpeg_encode *)attributes;
  jpeg_status = tj3Set (state->tj_handle, TJPARAM_QUALITY, jpeg_attr->quality);
  if (jpeg_status != 0)
    {
      POCL_MSG_ERR ("Could not set JPEG quality: %s (fatal? %d).\n",
                    tj3GetErrorStr (state->tj_handle),
                    tj3GetErrorCode (state->tj_handle));
      *status = CL_OUT_OF_RESOURCES;
      return NULL;
    }

  /* TODO: should this be configurable from the attributes? */
  jpeg_status = tj3Set (state->tj_handle, TJPARAM_SUBSAMP, TJSAMP_420);
  if (jpeg_status != 0)
    {
      POCL_MSG_ERR2 ("Could not set JPEG subsampling.", "%s",
                     tj3GetErrorStr (state->tj_handle));
      *status = CL_OUT_OF_RESOURCES;
      return NULL;
    }
  /* TODO: set JPEG no_realloc. */

  return state;
}

int
pocl_cpu_destroy_dbk_khr_jpeg_encode (void **kernel_data)
{
  jpeg_encode_state_t *state = (jpeg_encode_state_t *)*kernel_data;

  tj3Destroy (state->tj_handle);
  tj3Free (state->tj_buffer);
  POCL_MEM_FREE (state);
  return CL_SUCCESS;
}

int
pocl_cpu_execute_dbk_khr_jpeg_decode (cl_program program,
                                      cl_kernel kernel,
                                      pocl_kernel_metadata_t *meta,
                                      cl_uint dev_i,
                                      struct pocl_argument *arguments)
{
  cl_device_id dev = program->devices[dev_i];
  unsigned mem_id = dev->global_mem_id;
  void *jpeg = pocl_cpu_get_ptr (&arguments[0], mem_id);
  uint64_t *jpeg_size = pocl_cpu_get_ptr (&arguments[1], mem_id);
  void *image = pocl_cpu_get_ptr (&arguments[2], mem_id);

  /* make sure size is valid */
  assert (*jpeg_size > 0);

  jpeg_encode_state_t *state = kernel->data[dev_i];

  int status;
  status = tj3DecompressHeader (state->tj_handle, jpeg, *jpeg_size);
  if (status != 0)
    {
      POCL_MSG_ERR ("tj3DecompressHeader failed: %s (fatal? %d).\n",
                    tj3GetErrorStr (state->tj_handle),
                    tj3GetErrorCode (state->tj_handle));
      return CL_OUT_OF_RESOURCES;
    }

  size_t height = tj3Get (state->tj_handle, TJPARAM_JPEGHEIGHT);
  size_t width = tj3Get (state->tj_handle, TJPARAM_JPEGWIDTH);
  cl_mem jpeg_mem = *(cl_mem *)(arguments[2].value);
  if (jpeg_mem->size < width * height * tjPixelSize[TJCS_RGB])
    {
      POCL_MSG_ERR ("pocl_cpu_execute_dbk_khr_jpeg_decode: destination mem_obj"
                    " does not fit decoded RGB result.\n");
      assert (0 && "improper parameters");
      return CL_OUT_OF_RESOURCES;
    }

  status
    = tj3Decompress8 (state->tj_handle, jpeg, *jpeg_size, image, 0, TJCS_RGB);
  if (status == 0)
    return status;
  POCL_MSG_ERR ("tj3Decompress8 failed: %s (fatal? %d).\n",
                tj3GetErrorStr (state->tj_handle),
                tj3GetErrorCode (state->tj_handle));
  return CL_OUT_OF_RESOURCES;
}

void *
pocl_cpu_init_dbk_khr_jpeg_decode (void const *attributes, int *status)
{
  jpeg_decode_state_t *state
    = (jpeg_decode_state_t *)calloc (1, sizeof (jpeg_decode_state_t));
  state->tj_handle = tj3Init (TJINIT_DECOMPRESS);
  if (state->tj_handle == NULL)
    {
      POCL_MSG_ERR ("create decompress tj_handle failed: %s (fatal? %d).\n",
                    tj3GetErrorStr (state->tj_handle),
                    tj3GetErrorCode (state->tj_handle));
      *status = CL_OUT_OF_RESOURCES;
      return NULL;
    }

  return state;
}

int
pocl_cpu_destroy_dbk_khr_jpeg_decode (void **kernel_data)
{
  jpeg_decode_state_t *state = (jpeg_decode_state_t *)*kernel_data;
  tj3Destroy (state->tj_handle);
  POCL_MEM_FREE (state);
  return CL_SUCCESS;
}

#endif
