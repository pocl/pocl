/* OpenCL runtime library: Dynalib library utility functions implemented
   using POSIX <dlfcn.h>

   Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_dynlib.h"

#ifdef HAVE_DLFCN_H
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#include <dlfcn.h>
#endif

void *
pocl_dynlib_open (const char *path, int lazy, int local)
{
  int flags = 0;
  if (lazy)
    flags |= RTLD_LAZY;
  else
    flags |= RTLD_NOW;

  if (local)
    flags |= RTLD_LOCAL;
  else
    flags |= RTLD_GLOBAL;

  void *handle = dlopen (path, flags);
  if (handle == NULL)
    {
      char *err_msg = dlerror ();
      if (err_msg == NULL)
        POCL_MSG_ERR ("dlopen() failed without an error message\n");
      else
        POCL_MSG_ERR ("dlopen() error: %s\n", err_msg);
    }
  return handle;
}

int
pocl_dynlib_close (void *dynlib_handle)
{
  return dlclose (dynlib_handle);
}

void *
pocl_dynlib_symbol_address (void *dynlib_handle, const char *symbol_name)
{
  void *addr = dlsym (dynlib_handle, symbol_name);
  if (addr == NULL)
    {
      char *err_msg = dlerror ();
      if (err_msg == NULL)
        POCL_MSG_ERR ("dlsym() failed without an error message\n");
      else
        POCL_MSG_ERR ("dlsym() error: %s\n", err_msg);
    }
  return addr;
}

const char *
pocl_dynlib_pathname (void *address)
{
  Dl_info info;
  info.dli_fname = NULL;

  if (!dladdr (address, &info) || info.dli_fname == NULL)
    POCL_MSG_ERR ("dladdr() returned an error\n");
  return info.dli_fname;
}
