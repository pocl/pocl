/* pocl.h - global pocl declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#ifndef POCL_H
#define POCL_H

#include <CL/opencl.h>

#define POCL_ADDRESS_SPACE_PRIVATE 0
#define POCL_ADDRESS_SPACE_GLOBAL 3
#define POCL_ADDRESS_SPACE_LOCAL 4
#define POCL_ADDRESS_SPACE_CONSTANT 5

struct pocl_context {
  cl_uint work_dim;
  cl_uint num_groups[3];
  cl_uint group_id[3];
  cl_uint global_offset[3];
};

typedef void (*pocl_workgroup) (void **, struct pocl_context *);

#endif /* POCL_H */
