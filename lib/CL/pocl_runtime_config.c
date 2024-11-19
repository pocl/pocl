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

#ifdef __WIN32__

#include <windows.h>

// On Windows, we need to make sure to use the Unicode-version of the
// environment functions, as otherwise we may may not pick up on all
// environment variables. This helper checks both versions.
static const char *
getenv_helper (const char *key)
{
  // Convert key to wide char
  size_t len = strlen (key) + 1;
  wchar_t *wkey = (wchar_t *)malloc (len * sizeof (wchar_t));
  if (!wkey)
    {
      return NULL;
    }
  mbstowcs (wkey, key, len);

  unsigned long size = GetEnvironmentVariableW (wkey, NULL, 0);
  if (size == 0)
    {
      free (wkey);
      return NULL;
    }

  wchar_t *wval = (wchar_t *)malloc (size * sizeof (wchar_t));
  if (!wval)
    {
      free (wkey);
      return NULL;
    }

  if (GetEnvironmentVariableW (wkey, wval, size) == 0)
    {
      free (wkey);
      free (wval);
      return NULL;
    }

  free (wkey);

  size_t needed = wcstombs (NULL, wval, 0) + 1;
  char *result = (char *)malloc (needed);
  if (!result)
    {
      free (wval);
      return NULL;
    }

  wcstombs (result, wval, needed);
  free (wval);
  return result; // Still leaks to match getenv behavior
}
#else
#define getenv_helper getenv
#endif

/* Can be used to query if the given option has been set by the user. */
int pocl_is_option_set(const char *key)
{
  return getenv_helper (key) != NULL ? 1 : 0;
}

/* Returns an integer value for the option with the given key string. */
int pocl_get_int_option(const char *key, int default_value)
{
  const char *val = getenv_helper (key);
  return val ? atoi (val) : default_value;
}

/* Returns a boolean value for the option with the given key string. */
int
pocl_get_bool_option (const char *key, int default_value)
{
  const char *val = getenv_helper (key);
  if (val != NULL)
    return (strncmp (val, "1", 1) == 0);
  return default_value;
}

const char *
pocl_get_string_option (const char *key, const char *default_value)
{
  const char *val = getenv_helper (key);
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
  const char *val = getenv_helper (key);
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
