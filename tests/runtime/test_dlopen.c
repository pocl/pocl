/* Test that pocl libraries can be dlopen()ed

   Copyright (c) 2021 pocl developers

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

#include <dlfcn.h>
#include <stdio.h>

#ifdef __APPLE__
#define SHLIB_EXT "dylib"
#else
#define SHLIB_EXT "so"
#endif

int
main (int argc, char **argv)
{
  int ret = 0;
  const char *libpocl = "$ORIGIN/../../lib/CL/libpocl." SHLIB_EXT;
  char libdevice[4096] = "";
  if (argc > 1)
    snprintf (libdevice, sizeof (libdevice),
              "$ORIGIN/../../lib/CL/devices/%s/libpocl-devices-%s." SHLIB_EXT, argv[1],
              argv[1]);

  void *handle_libpocl = dlopen (libpocl, RTLD_NOW | RTLD_GLOBAL);
  if (!handle_libpocl)
    {
      fprintf (stderr, "dlopen(%s, RTLD_NOW | RTLD_GLOBAL) failed: %s\n",
               libpocl, dlerror ());
      ret = 1;
    }

  if (ret == 0 && argc > 1)
    {
      void *handle_device = dlopen (libdevice, RTLD_NOW);
      if (!handle_device)
        {
          fprintf (stderr, "dlopen(%s, RTLD_NOW) failed: %s\n", libdevice,
                   dlerror ());
          ret = 1;
        }
      if (handle_device)
        dlclose (handle_device);
    }

  if (handle_libpocl)
    dlclose (handle_libpocl);

  return ret;
}
