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
#include "pthread/pocl-pthread.h"

#if defined(BUILD_SPU)
#include "cellspu/cellspu.h"
#endif

#if defined(TCE_AVAILABLE)
#include "tce/ttasim/ttasim.h"
#endif

/* the enabled devices */
struct _cl_device_id* pocl_devices = NULL;
int pocl_num_devices = 0;

/* Init function prototype */
typedef void (*init_device_ops)(struct pocl_device_ops*);

/* All init function for device operations available to pocl */
static init_device_ops pocl_devices_init_ops[] = {
  pocl_pthread_init_device_ops,
  pocl_basic_init_device_ops,
#if defined(BUILD_SPU)
  pocl_cellspu_init_device_ops,
#endif
#if defined(TCE_AVAILABLE)
  pocl_ttasim_init_device_ops,
#endif
};

#define POCL_NUM_DEVICE_TYPES (sizeof(pocl_devices_init_ops) / sizeof((pocl_devices_init_ops)[0]))

static struct pocl_device_ops pocl_device_ops[POCL_NUM_DEVICE_TYPES] = {0};

static inline void
pocl_device_common_init(struct _cl_device_id* dev)
{
  POCL_INIT_OBJECT(dev);
  dev->driver_version = PACKAGE_VERSION;
  if(dev->version == NULL)
    dev->version = "OpenCL 1.2 pocl";

  dev->short_name = strdup(dev->ops->device_name);
  if(dev->long_name == NULL)
    dev->long_name = dev->short_name;
}

void 
pocl_init_devices()
{
  const char *device_list;
  char *ptr, *tofree, *token, *saveptr;
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
  
  ptr = tofree = strdup(device_list);
  while ((token = strtok_r (ptr, " ", &saveptr)) != NULL)
    {
      ++pocl_num_devices;
      ptr = NULL;
    }
  free (tofree);
  
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      pocl_devices_init_ops[i](&pocl_device_ops[i]);
      assert(pocl_device_ops[i].device_name != NULL);
    }

  pocl_devices = calloc (pocl_num_devices, sizeof *pocl_devices);

  ptr = tofree = strdup(device_list);
  devcount = 0;
  while ((token = strtok_r (ptr, " ", &saveptr)) != NULL)
    {
      char found = 0;
      for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
        {
          if (strcmp(pocl_device_ops[i].device_name, token) == 0)
            {
              /* Check if there are device-specific parameters set in the
                 POCL_DEVICEn_PARAMETERS env. */
              char env_name[1024];
              
              if (snprintf (env_name, 1024, "POCL_DEVICE%d_PARAMETERS", devcount) < 0)
                POCL_ABORT("Unable to generate the env string.");
              pocl_devices[devcount].ops = &pocl_device_ops[i];
              /* The default value for the global memory space identifier is
                 the same as the device id. The device instance can then override 
                 it to point to some other device's global memory id in case of
                 a shared global memory. */
              pocl_devices[devcount].global_mem_id = devcount;
              assert(pocl_device_ops[i].init_device_infos);
              pocl_device_ops[i].init_device_infos(&pocl_devices[devcount]);
              pocl_device_common_init(&pocl_devices[devcount]);
              assert(pocl_device_ops[i].init);
              pocl_device_ops[i].init(&pocl_devices[devcount], getenv(env_name));
              
              pocl_devices[devcount].dev_id = devcount;
              devcount++;
              found = 1;
              break;
            }
        }
      if (!found) 
          POCL_ABORT("device type not found\n");
      ptr = NULL;
    }
  free (tofree);
}
