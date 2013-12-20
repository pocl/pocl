/* pocl_runtime_config.c: functions to query pocl runtime configuration settings

   Copyright (c) 2013 Pekka Jääskeläinen
   
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

#include "pocl_runtime_config.h"
#include <stdlib.h>
#include <string.h>

/* TODO: cache the options and to avoid calling getenv more than once 
   per option. */

/* Can be used to query if the given option has been set by the user. */
int pocl_is_option_set(const char *key)
{
  return getenv(key) != NULL;
}

/* Returns an integer value for the option with the given key string. */
int pocl_get_int_option(const char *key, int default_value)
{
  const char* val = getenv(key);
  if (val == NULL) return default_value;
  return atoi(val);
}

/* Returns a boolean value for the option with the given key string. */
int pocl_get_bool_option(const char *key, int default_value) 
{
  const char* val = getenv(key);
  if (val == NULL) return default_value;
  return strncmp(val, "1", 1) == 0;
}

const char* pocl_get_string_option(const char *key, const char *default_value) 
{
  const char* val = getenv(key); 
  if (val == NULL) return default_value;
  return val;
}
