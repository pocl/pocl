/* pocl_runtime_config.h: functions to query pocl runtime configuration
   settings

   Copyright (c) 2013 Pekka Jääskeläinen

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

#ifndef _POCL_RUNTIME_CONFIG_H
#define _POCL_RUNTIME_CONFIG_H

#include "pocl_export.h"

#ifdef __cplusplus
extern "C" {
#endif

POCL_EXPORT
int pocl_is_option_set(const char *key);
POCL_EXPORT
int pocl_get_int_option(const char *key, int default_value);
POCL_EXPORT
int pocl_get_bool_option(const char *key, int default_value);
POCL_EXPORT
const char* pocl_get_string_option(const char *key, const char *default_value);

/**
 * \brief Gets a path as specified in the environment, or a default value.
 *
 * This function retrieves the user-specified path to an executable `name` as
 * specified in the environment variable `POCL_PATH_<name>`. If the environment
 * variable is not set, the function returns the provided default value.
 *
 * \param name The name of the path being queried. This is used to construct
 * the environment variable name.
 *
 * \param default_value The default value to return if the environment variable
 * is not set.
 *
 * \return A pointer to a string containing the path, either from the
 * environment variable or the default value.
 */
POCL_EXPORT
const char *pocl_get_path (const char *name, const char *default_value);

/**
 * \brief Gets environment-specified arguments for invoking a binary.
 *
 * This function retrieves user-specified arguments from the
 * semicolon-separated environment variable `POCL_ARGS_<name>`, that should
 * also be passed when invoking the executable `name`. If the environment
 * variable is not set, the function returns NULL and sets `n` to zero.
 * Otherwise, `n` is set to the number of arguments found in the string, and
 * the function returns a pointer to a sequence of null-terminated strings.
 *
 * \param name The name of the executable for which the arguments are intended.
 * This is used to construct the environment variable name.
 *
 * \param n Pointer to an integer where the number of arguments will be stored.
 *
 * \return A pointer to a sequence of null-terminated strings containing the
 * arguments, or NULL if the environment variable is not set.
 *
 * \note The returned string should be freed by the caller to avoid memory
 * leaks.
 */
POCL_EXPORT
char *pocl_get_args (const char *name, int *n);

#ifdef __cplusplus
}
#endif


#endif
