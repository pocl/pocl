/* pocl_dbk_khr_onnxrt_shared.c - Defined Built-in Kernels interfaces.

   Copyright (c) 2024 Jan Solanti <jan.solanti@tuni.fi>

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
#include "pocl_dbk_khr_onnxrt_shared.h"
#include "pocl_tensor_util.h"

/** Construct a deep copy of the given attributes */
cl_dbk_attributes_khr_onnx_inference *
pocl_copy_onnx_inference_dbk_attributes (
    const cl_dbk_attributes_khr_onnx_inference *src)
{
  int err;
  cl_dbk_attributes_khr_onnx_inference *attrs
      = malloc (sizeof (cl_dbk_attributes_khr_onnx_inference));
  if (attrs == NULL)
    return NULL;
  attrs->model_size = src->model_size;
  attrs->num_inputs = src->num_inputs;
  attrs->num_outputs = src->num_outputs;
  attrs->num_initializers = src->num_initializers;
  attrs->model_data = malloc (attrs->model_size);
  if (attrs->model_data == NULL)
    goto ERROR;
  memcpy ((char *)attrs->model_data, src->model_data, attrs->model_size);

  /******** Copy input tensor data ********/
  attrs->input_tensor_names = calloc (attrs->num_inputs, sizeof (char *));
  if (attrs->input_tensor_names == NULL)
    goto ERROR;
  attrs->input_tensor_descs
      = calloc (attrs->num_inputs, sizeof (cl_tensor_desc));
  if (attrs->input_tensor_descs == NULL)
    goto ERROR;
  for (size_t i = 0; i < attrs->num_inputs; ++i)
    {
      attrs->input_tensor_names[i] = strdup (src->input_tensor_names[i]);
      if (attrs->input_tensor_names[i] == NULL)
        goto ERROR;
      memcpy ((cl_tensor_desc *)&attrs->input_tensor_descs[i],
              &src->input_tensor_descs[i], sizeof (cl_tensor_desc));
      err = pocl_copy_tensor_desc_layout (
        (cl_tensor_desc *)&attrs->input_tensor_descs[i],
        &src->input_tensor_descs[i]);
      if (err != CL_SUCCESS)
        goto ERROR;
    }

  /******** Copy output tensor data ********/
  attrs->output_tensor_names = calloc (attrs->num_outputs, sizeof (char *));
  if (attrs->output_tensor_names == NULL)
    goto ERROR;
  attrs->output_tensor_descs
      = calloc (attrs->num_outputs, sizeof (cl_tensor_desc));
  if (attrs->output_tensor_descs == NULL)
    goto ERROR;
  for (size_t i = 0; i < attrs->num_outputs; ++i)
    {
      attrs->output_tensor_names[i] = strdup (src->output_tensor_names[i]);
      if (attrs->output_tensor_names[i] == NULL)
        goto ERROR;
      memcpy ((cl_tensor_desc *)&attrs->output_tensor_descs[i],
              &src->output_tensor_descs[i], sizeof (cl_tensor_desc));
      err = pocl_copy_tensor_desc_layout (
        (cl_tensor_desc *)&attrs->output_tensor_descs[i],
        &src->output_tensor_descs[i]);
      if (err != CL_SUCCESS)
        goto ERROR;
    }

  /******** Copy initializer tensor data ********/
  if (attrs->num_initializers != 0)
    {
      attrs->initializer_names
          = calloc (attrs->num_initializers, sizeof (char *));
      if (attrs->initializer_names == NULL)
        goto ERROR;
      attrs->initializer_data
          = calloc (attrs->num_initializers, sizeof (char *));
      if (attrs->initializer_data == NULL)
        goto ERROR;
      attrs->initializer_tensor_descs
          = calloc (attrs->num_initializers, sizeof (cl_tensor_desc));
      if (attrs->initializer_tensor_descs == NULL)
        goto ERROR;
      for (size_t i = 0; i < attrs->num_initializers; ++i)
        {
          attrs->initializer_names[i] = strdup (src->initializer_names[i]);
          if (attrs->initializer_names[i] == NULL)
            goto ERROR;
          memcpy ((cl_tensor_desc *)&attrs->initializer_tensor_descs[i],
                  &src->initializer_tensor_descs[i], sizeof (cl_tensor_desc));
          err = pocl_copy_tensor_desc_layout (
            (cl_tensor_desc *)&attrs->initializer_tensor_descs[i],
            &src->initializer_tensor_descs[i]);
          if (err != CL_SUCCESS)
            goto ERROR;

          size_t data_len
              = pocl_tensor_type_size (attrs->initializer_tensor_descs[i].dtype);
          for (size_t dim = 0; dim < attrs->initializer_tensor_descs[i].rank;
               ++dim)
            {
              data_len *= attrs->initializer_tensor_descs[i].shape[dim];
            }
          attrs->initializer_data[i] = malloc (data_len);
          memcpy ((char *)attrs->initializer_data[i], src->initializer_data[i],
                  data_len);
        }
    }

  return attrs;

  ERROR:
  pocl_release_onnx_inference_dbk_attributes (attrs);
  return NULL;
}

/** Release kernel attributes. MUST NOT be called on user-provided attrs - only
 * on deep copies - as the user may have provided pointers to stack variables
 * or string literals. */
void
pocl_release_onnx_inference_dbk_attributes (
    cl_dbk_attributes_khr_onnx_inference *attrs)
{
  free ((char *)attrs->model_data);

  for (size_t i = 0; i < attrs->num_inputs; ++i)
    {
      if (attrs->input_tensor_descs)
        {
          free (attrs->input_tensor_descs[i].layout);
          memset ((cl_tensor_desc *)&(attrs->input_tensor_descs[i]), 0,
                  sizeof (cl_tensor_desc));
        }

      if (attrs->input_tensor_names)
        {
          free ((char *)attrs->input_tensor_names[i]);
          memset (&attrs->input_tensor_names[i], 0, sizeof (char *));
        }
    }
  free ((cl_tensor_desc *)attrs->input_tensor_descs);
  POCL_MEM_FREE (attrs->input_tensor_names);

  for (size_t i = 0; i < attrs->num_outputs; ++i)
    {
      if (attrs->output_tensor_descs)
        {
          free (attrs->output_tensor_descs[i].layout);
          memset ((cl_tensor_desc *)&(attrs->output_tensor_descs[i]), 0,
                  sizeof (cl_tensor_desc));
        }

      if (attrs->output_tensor_names)
        {
          free ((char *)attrs->output_tensor_names[i]);
          memset (&attrs->output_tensor_names[i], 0, sizeof (char *));
        }
    }
  free ((cl_tensor_desc *)attrs->output_tensor_descs);
  POCL_MEM_FREE (attrs->output_tensor_names);

  if (attrs->num_initializers != 0)
    {
      for (size_t i = 0; i < attrs->num_initializers; ++i)
        {
          free ((char *)attrs->initializer_names[i]);
          memset (&attrs->initializer_names[i], 0, sizeof (char *));

          free (attrs->initializer_tensor_descs[i].layout);
          memset ((cl_tensor_desc *)&(attrs->initializer_tensor_descs[i]), 0,
                  sizeof (cl_tensor_desc));

          free ((char *)attrs->initializer_data[i]);
          memset (&attrs->initializer_data[i], 0, sizeof (char *));
        }
    }
  POCL_MEM_FREE (attrs->initializer_names);
  free ((cl_tensor_desc *)attrs->initializer_tensor_descs);
  POCL_MEM_FREE (attrs->initializer_data);

  memset (attrs, 0, sizeof (cl_dbk_attributes_khr_onnx_inference));
  POCL_MEM_FREE (attrs);
}
