/* preprocess.h - client-side preprocessing for the remote driver

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, and/or copy the Software,
   and to permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

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

#ifndef POCL_REMOTE_PREPROCESS_H
#define POCL_REMOTE_PREPROCESS_H

#include "pocl_cl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runs a C preprocessor pass over the program source on the client side so
   that #include directives are resolved using the client's files instead of
   the (usually nonexistent) files on the remote server.

   On success, CL_SUCCESS is returned and *out_source is set to a malloc()ed,
   NUL-terminated string with the preprocessed source. On error,
   CL_BUILD_PROGRAM_FAILURE is returned and *err_msg is set to a malloc()ed
   human-readable message; both are NULL otherwise. */
int pocl_remote_preprocess_source(cl_program program, cl_device_id device,
                                  cl_uint num_input_headers,
                                  const cl_program *input_headers,
                                  const char **header_include_names,
                                  char **out_source, char **err_msg);

#ifdef __cplusplus
}
#endif

#endif
