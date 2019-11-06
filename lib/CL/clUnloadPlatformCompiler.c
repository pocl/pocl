/* OpenCL runtime library: clUnloadPlatformCompiler()
 *
 *    Copyright (c) 2016 Tom Gall Tampere Univ. of Tech.
 *       
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *                            
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *                               
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *    THE SOFTWARE.
 **/

#include "pocl_cl.h"
#include "pocl_llvm.h"
#include "pocl_shared.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clUnloadPlatformCompiler)(cl_platform_id platform)
CL_API_SUFFIX__VERSION_1_2
{
#if defined(ENABLE_LLVM)
  cl_platform_id pocl_id;
  POname (clGetPlatformIDs) (1, &pocl_id, NULL);
  if (platform != pocl_id)
    {
      POCL_MSG_WARN (
          "clUnloadPlatformCompiler called with non-pocl platform! \n");
      return CL_INVALID_PLATFORM;
    }
#else
  POCL_MSG_WARN (
      "clUnloadPlatformCompiler called with LLVM-less build of pocl! \n");
#endif
  pocl_check_uninit_devices ();
  return CL_SUCCESS;
}
POsym(clUnloadPlatformCompiler)
