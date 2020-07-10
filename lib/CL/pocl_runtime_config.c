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
#include "utlist.h"
#include "pocl_cl.h"

#include <stdlib.h>
#include <string.h>

typedef struct env_data env_data;
struct env_data
{
  char *env;
  char *value;
  env_data *next;
};

static env_data *volatile env_cache = 0;
static pocl_lock_t lock = POCL_LOCK_INITIALIZER;

static env_data* find_env (env_data* cache, const char* key)
{
  env_data* ed;
  char *value;

  POCL_LOCK(lock);
  LL_FOREACH(cache, ed)
    {
      if (strcmp(ed->env, key) == 0)
        {
          POCL_UNLOCK(lock);
          return ed;
        }
    }
  if ((value = getenv(key)))
    {
      ed = (env_data*) malloc (sizeof (env_data));
      ed->env = strdup (key);
      ed->value = strdup (value);
      ed->next = NULL;
      LL_APPEND(env_cache, ed);
      POCL_UNLOCK(lock);
      return ed;
    }
  
  POCL_UNLOCK(lock);
  return NULL;
}
/* Can be used to query if the given option has been set by the user. */
int pocl_is_option_set(const char *key)
{
  env_data* ed = NULL;
  return (ed = find_env (env_cache, key)) ? 1 : 0;
}

/* Returns an integer value for the option with the given key string. */
int pocl_get_int_option(const char *key, int default_value)
{
  env_data *ed;
  return (ed = find_env (env_cache, key)) ? atoi(ed->value) : default_value;
}

/* Returns a boolean value for the option with the given key string. */
int pocl_get_bool_option(const char *key, int default_value) 
{
  env_data *ed;
  if ((ed = find_env(env_cache, key)))
    return (strncmp(ed->value, "1", 1) == 0);
  return default_value;
}
 
const char* pocl_get_string_option(const char *key, const char *default_value) 
{
  env_data *ed;
  return (ed = find_env (env_cache, key)) ? ed->value : default_value;
}
