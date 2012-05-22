/* Definition of available OpenCL devices.

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include <unistd.h>
#include <string.h>

#include "devices.h"
#include "common.h"
#include "basic/basic.h"
#include "pthread/pthread.h"

#if defined(TCE_AVAILABLE)
#include "ttasim/ttasim.h"
#endif

/* the enabled devices */
struct _cl_device_id* pocl_devices = NULL;
int pocl_num_devices = 0;

#ifdef TCE_AVAILABLE
#define POCL_NUM_DEVICE_TYPES 3
#else
#define POCL_NUM_DEVICE_TYPES 2
#endif

/* all device types available to the pocl */
static struct _cl_device_id pocl_device_types[POCL_NUM_DEVICE_TYPES] = {
  POCL_DEVICES_PTHREAD,
  POCL_DEVICES_BASIC
#if defined(TCE_AVAILABLE)
    ,
    POCL_DEVICES_TTASIM
#endif
};

void 
pocl_init_devices()
{
  const char *device_list, *device_list2;
  char *token, *saveptr, *saveptr2, *ptr;
  int i, devcount;
  if (pocl_num_devices > 0)
    return;
  
  if (getenv(POCL_DEVICES_ENV) != NULL) 
    {
      device_list = getenv(POCL_DEVICES_ENV);
    }
  else
    {
      device_list = "pthread";
    }
  
  ptr = strdup(device_list);

  /* strtok_r is a braindead function that modifies the ptr so
     we have to copy the original for later use. */
  device_list2 = strdup (device_list);
  while ((token = strtok_r (ptr, " ", &saveptr)) != NULL)
    {
      ++pocl_num_devices;
      ptr = NULL;
    }

  pocl_devices = (struct _cl_device_id*) 
    malloc (sizeof(struct _cl_device_id) * pocl_num_devices);

  ptr = device_list2;

  devcount = 0;
  while ((token = strtok_r (ptr, " ", &saveptr2)) != NULL)
    {
      struct _cl_device_id* device_type = NULL;

      for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
        {
          if (strcmp(pocl_device_types[i].name, token) == 0)
            {
              /* Check if there are device-specific parameters set in the
                 POCL_DEVICEn_PARAMETERS env. */
              char env_name[1024];
              
              if (snprintf (env_name, 1024, "POCL_DEVICE%d_PARAMETERS", devcount) < 0)
                POCL_ABORT("Unable to generate the env string.");

              device_type = &pocl_device_types[i];
              memcpy (&pocl_devices[devcount], device_type, sizeof(struct _cl_device_id));
              pocl_devices[devcount].init(&pocl_devices[devcount], getenv(env_name));
              pocl_devices[devcount].dev_id = devcount;
              devcount++;
              break;
            }
        }
      if (device_type == NULL)
        POCL_ABORT("device type not found\n");
      ptr = NULL;
    }
  free (device_list2);
  free (ptr);
}
