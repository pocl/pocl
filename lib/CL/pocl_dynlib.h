/* OpenCL runtime library: Dynamic library utility functions

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

#ifndef POCL_DYNLIB_H
#define POCL_DYNLIB_H

#include "pocl_cl.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Opens the dynamic library in the given path.
 *
 * \param Path the full path to the dynlib.
 * \param lazy Set to 1, if wanting to evaluate the symbols lazily.
 * \param local Set to 1, in case the symbols should not be made
 *              visible to libraries loaded later.
 * \return An OS-specific handle to it, or NULL in case of an error.
 */
POCL_EXPORT
void *pocl_dynlib_open (const char *path, int lazy, int local);

/**
 * Closes the dynamic library in the given path.
 *
 * Reference counting is done to ensure the library is not unloaded too early
 * if there have been multiple opens of it.
 *
 * \return 1 on success, zero on an error.
 */
POCL_EXPORT
int pocl_dynlib_close (void *dynlib_handle);

/**
 * Returns the address of a symbol in the given dynamic library.
 *
 * \param dynlib_handle The handle of the dynamic library.
 * \param symbol_name The name of the symbol to resolve.
 * \return The address of the symbol, NULL on error.
 */
POCL_EXPORT
void *pocl_dynlib_symbol_address (void *dynlib_handle,
                                  const char *symbol_name);

/**
 * Returns the pathname of the library of where the given address was
 * loaded
 *
 * \return The pathname, NULL on error.
 */
POCL_EXPORT
const char *pocl_dynlib_pathname (void *address);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
