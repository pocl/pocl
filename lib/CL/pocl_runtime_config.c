/* pocl_runtime_config.c: functions to query pocl runtime configuration
   settings

   Copyright (c) 2013 Pekka Jääskeläinen / Tampere University (of Technology)
                 2023 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_runtime_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Can be used to query if the given option has been set by the user. */
int pocl_is_option_set(const char *key)
{
  return getenv (key) != NULL ? 1 : 0;
}

/* Returns an integer value for the option with the given key string. */
int pocl_get_int_option(const char *key, int default_value)
{
  const char *val = getenv (key);
  return val ? atoi (val) : default_value;
}

/* Returns a boolean value for the option with the given key string. */
int
pocl_get_bool_option (const char *key, int default_value)
{
  const char *val = getenv (key);
  if (val != NULL)
    return (strncmp (val, "1", 1) == 0);
  return default_value;
}

const char *
pocl_get_string_option (const char *key, const char *default_value)
{
  const char *val = getenv (key);
  return val != NULL ? val : default_value;
}

const char *
pocl_get_path (const char *name, const char *default_value)
{
  char key[256];
  snprintf (key, sizeof (key), "POCL_PATH_%s", name);
  return pocl_get_string_option (key, default_value);
}

char *
pocl_get_args (const char *name, int *n)
{
  char key[256];
  snprintf (key, sizeof (key), "POCL_ARGS_%s", name);
  const char *val = getenv (key);
  if (val == NULL)
    {
      *n = 0;
      return NULL;
    }

  char *args = strdup (val);
  *n = 1;
  for (char *p = args; *p; ++p)
    {
      if (*p == ';')
        {
          *p = 0;
          ++(*n);
        }
    }
  return args;
}
