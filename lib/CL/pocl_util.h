/* OpenCL runtime library: pocl_util utility functions

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifndef POCL_UTIL_H
#define POCL_UTIL_H

#include <stdint.h>

#pragma GCC visibility push(hidden)
#ifdef __cplusplus
extern "C" {
#endif

/* Create a temporary directory that is cleaned atexit.
 *
 * The returned path name is a string that will be alive
 * until the exit.
 */
char *pocl_create_temp_dir();
void remove_directory (const char *path_name);

uint32_t byteswap_uint32_t (uint32_t word, char should_swap);
float byteswap_float (float word, char should_swap);

#ifdef __cplusplus
}
#endif
#pragma GCC visibility pop

/* Common macro for cleaning up "*GetInfo" API call implementations.
 * All the *GetInfo functions have been specified to look alike, 
 * and have been implemented to use the same variable names, so this
 * code can be shared.
 */
#define POCL_RETURN_GETINFO(__TYPE__, __VALUE__)                \
  {                                                                 \
    size_t const value_size = sizeof(__TYPE__);                     \
    if (param_value)                                                \
      {                                                             \
        if (param_value_size < value_size) return CL_INVALID_VALUE; \
        *(__TYPE__*)param_value = __VALUE__;                        \
      }                                                             \
    if (param_value_size_ret)                                       \
      *param_value_size_ret = value_size;                           \
    return CL_SUCCESS;                                              \
  } 



#endif
