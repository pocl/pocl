/* OpenCL runtime library: clRetainSampler()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL POname (clRetainSampler) (cl_sampler sampler)
    CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (sampler)), CL_INVALID_SAMPLER);

  POCL_RETAIN_OBJECT (sampler);
  POCL_MSG_PRINT_REFCOUNTS ("RETAIN Sampler %p  : %d\n", sampler,
                            sampler->pocl_refcount);

  return CL_SUCCESS;
}

POsym(clRetainSampler)
