/* pocl_dbk_khr_dnn_utils.c - cpu implementation of neural network related DBKs.

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

#include "pocl_dbk_khr_dnn_utils.hh"
#include "common_utils.h"
#include <opencv2/dnn.hpp>
#include <string>

using namespace std;
using namespace cv::dnn;

int pocl_cpu_execute_dbk_khr_dnn_nms(cl_program program, cl_kernel kernel,
                                     pocl_kernel_metadata_t *meta,
                                     cl_uint dev_i,
                                     struct pocl_argument *arguments) {

  cl_device_id dev = program->devices[dev_i];
  cl_dbk_attributes_exp_dnn_nms *attributes =
      (cl_dbk_attributes_exp_dnn_nms *)meta->builtin_kernel_attrs;
  unsigned mem_id = dev->global_mem_id;
  int32_t *boxes =
      static_cast<int32_t *>(pocl_cpu_get_ptr(&arguments[0], mem_id));
  float *scores = static_cast<float *>(pocl_cpu_get_ptr(&arguments[1], mem_id));
  int32_t *index_count =
      static_cast<int32_t *>(pocl_cpu_get_ptr(&arguments[2], mem_id));
  int32_t *indices =
      static_cast<int32_t *>(pocl_cpu_get_ptr(&arguments[3], mem_id));

  vector<cv::Rect> boxVector;
  boxVector.reserve(attributes->num_boxes);
  cv::Rect rect;
  for (int i = 0; i < attributes->num_boxes * 4; i += 4) {
    boxVector.emplace_back(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]);
  }
  vector<float> scoreVector(scores, scores + attributes->num_boxes);

  vector<int32_t> indexVector;
  NMSBoxes(boxVector, scoreVector, attributes->score_threshold,
           attributes->nms_threshold, indexVector);

  indexVector.resize(MIN(attributes->top_k, indexVector.size()));

  // size of indices is already checked in setkernelarg to make sure it fits
  memcpy(indices, indexVector.data(), indexVector.size() * sizeof(int32_t));
  *index_count = indexVector.size();

  return CL_SUCCESS;
}