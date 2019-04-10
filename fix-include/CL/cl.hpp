/* cl.hpp - fix for the Khronos Group header

   Copyright (c) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>

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

// fix problems with current khronos cl.hpp header file
// khronos cl.hpp is a 1.1 OpenCL header file so it needs 1.1 declaration
// defining CL_USE_DEPRECATED_OPENCL_1_1_APIS allow the compilation
// however, this generates warnings
// pragmas are here to avoid the warnings
// but g++ has a bug for now: http://bug.debian.org/686178

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if defined __clang__

#  pragma clang diagnostic push
#    pragma clang diagnostic ignored "-W#warnings"
#    include_next <CL/cl.hpp>
#  pragma clang diagnostic pop

#elif defined GCC_VERSION && GCC_VERSION >= 40600

#  pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcpp"
#    include_next <CL/cl.hpp>
#  pragma GCC diagnostic pop

#else

#  include_next <CL/cl.hpp>

#endif
