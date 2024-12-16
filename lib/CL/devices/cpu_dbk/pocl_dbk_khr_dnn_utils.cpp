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
#include "pocl_mem_management.h"
#include <opencv2/dnn.hpp>
#include <string>

using namespace std;
using namespace cv::dnn;

int pocl_cpu_execute_dbk_khr_nms_box(cl_program program, cl_kernel kernel,
                                     pocl_kernel_metadata_t *meta,
                                     cl_uint dev_i,
                                     struct pocl_argument *arguments) {

  cl_device_id Dev = program->devices[dev_i];
  cl_dbk_attributes_nms_box_exp *attributes =
      (cl_dbk_attributes_nms_box_exp *)meta->builtin_kernel_attrs;
  unsigned MemId = Dev->global_mem_id;
  int32_t *Boxes =
      static_cast<int32_t *>(pocl_cpu_get_ptr(&arguments[0], MemId));
  float *Scores = static_cast<float *>(pocl_cpu_get_ptr(&arguments[1], MemId));
  int32_t *IndexCount =
      static_cast<int32_t *>(pocl_cpu_get_ptr(&arguments[2], MemId));
  int32_t *Indices =
      static_cast<int32_t *>(pocl_cpu_get_ptr(&arguments[3], MemId));

  vector<cv::Rect> BoxVector;
  BoxVector.reserve(attributes->num_boxes);
  cv::Rect rect;
  for (size_t i = 0; i < attributes->num_boxes * 4; i += 4)
    BoxVector.emplace_back(Boxes[i], Boxes[i + 1], Boxes[i + 2], Boxes[i + 3]);

  vector<float> scoreVector(Scores, Scores + attributes->num_boxes);

  vector<int32_t> indexVector;
  NMSBoxes(BoxVector, scoreVector, attributes->score_threshold,
           attributes->iou_threshold, indexVector);

  indexVector.resize(std::min<size_t>(attributes->top_k, indexVector.size()));

  // size of Indices is already checked in setkernelarg to make sure it fits
  memcpy(Indices, indexVector.data(), indexVector.size() * sizeof(int32_t));
  *IndexCount = indexVector.size();

  return CL_SUCCESS;
}