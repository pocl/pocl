/* OpenCL runtime library: clCompileProgram()

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

#include "pocl_cl.h"
#include "pocl_shared.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clCompileProgram) (cl_program program,
                           cl_uint num_devices,
                           const cl_device_id *device_list,
                           const char *options,
                           cl_uint num_input_headers,
                           const cl_program *input_headers,
                           const char **header_include_names,
                           void (CL_CALLBACK *pfn_notify) (cl_program program,
                                                         void *user_data),
                           void *user_data)
CL_API_SUFFIX__VERSION_1_2
{
  return compile_and_link_program (1, 0, program,
                                   num_devices, device_list, options,
                                   num_input_headers, input_headers,
                                   header_include_names, 0, NULL,
                                   pfn_notify, user_data);
}
POsym (clCompileProgram)
