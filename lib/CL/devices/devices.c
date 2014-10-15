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
#include <ctype.h>

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

#define MAX_DEV_NAME_LEN 64

/* the enabled devices */
static struct _cl_device_id* pocl_devices = NULL;
unsigned int pocl_num_devices = 0;

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

/**
 * Get the number of specified devices from environnement
 */
int pocl_device_get_env_count(const char *dev_type)
{
  const char *dev_env = getenv(POCL_DEVICES_ENV);
  char *ptr, *saveptr, *tofree, *token;
  unsigned int dev_count = 0;
  if (dev_env == NULL) 
    {
      return -1;
    }
  ptr = tofree = strdup(dev_env);
  while ((token = strtok_r (ptr, " ", &saveptr)) != NULL)
    {
      if(strcmp(token, dev_type) == 0)
        dev_count++;
      ptr = NULL;
    }
  POCL_MEM_FREE(tofree);

  return dev_count;
}

static unsigned int __total_dev_count = 0;

unsigned int
pocl_get_devices(cl_device_type device_type, struct _cl_device_id **devices, unsigned int num_devices)
{
  struct _cl_device_id *device;
  unsigned int i, dev_added = 0;

  for (i = 0; i < pocl_num_devices; ++i)
    {
      if ((pocl_devices[i].type & device_type) &&
          (pocl_devices[i].available == CL_TRUE))
        {
            if (dev_added < num_devices)
              {
                devices[dev_added] = &pocl_devices[i];
                ++dev_added;
              }
            else
              {
                break;
              }
        }
    }
  return dev_added;
}

unsigned int
pocl_get_device_type_count(cl_device_type device_type)
{
  int count = 0;
  unsigned int i;

  for (i = 0; i < pocl_num_devices; ++i)
    {
      if ((pocl_devices[i].type & device_type) &&
          (pocl_devices[i].available == CL_TRUE))
        {
           ++count;
        }
    }
  return count;
}


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

static inline void
str_toupper(char *out, const char *in)
{
  int i;

  for (i = 0; in[i] != '\0'; i++)
    out[i] = toupper(in[i]);
  out[i] = '\0';
}

void 
pocl_init_devices()
{
  static unsigned int init_done = 0;
  static pocl_lock_t pocl_init_lock = POCL_LOCK_INITIALIZER;

  int i, j, dev_index;
  char env_name[1024];
  char dev_name[MAX_DEV_NAME_LEN] = {0};
  unsigned int device_count[POCL_NUM_DEVICE_TYPES];

  if (init_done == 0)
    POCL_INIT_LOCK(pocl_init_lock);
  POCL_LOCK(pocl_init_lock);
  if (init_done) 
    {
      POCL_UNLOCK(pocl_init_lock);
      return;
    }

  /* Init operations */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      pocl_devices_init_ops[i](&pocl_device_ops[i]);
      assert(pocl_device_ops[i].device_name != NULL);

      /* Probe and add the result to the number of probbed devices */
      assert(pocl_device_ops[i].probe);
      device_count[i] = pocl_device_ops[i].probe(&pocl_device_ops[i]);
      pocl_num_devices += device_count[i];
    }

  assert(pocl_num_devices > 0);
  pocl_devices = calloc(pocl_num_devices, sizeof(struct _cl_device_id));
  if (pocl_devices == NULL)
    POCL_ABORT("Can not allocate memory for devices\n");

  dev_index = 0;
  /* Init infos for each probbed devices */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      assert(pocl_device_ops[i].init);
      for (j = 0; j < device_count[i]; ++j)
        {
          pocl_devices[dev_index].ops = &pocl_device_ops[i];
          /* The default value for the global memory space identifier is
             the same as the device id. The device instance can then override 
             it to point to some other device's global memory id in case of
             a shared global memory. */
          pocl_devices[dev_index].global_mem_id = dev_index;
          
          pocl_device_ops[i].init_device_infos(&pocl_devices[dev_index]);

          pocl_device_common_init(&pocl_devices[dev_index]);

          str_toupper(dev_name, pocl_device_ops[i].device_name);
          /* Check if there are device-specific parameters set in the
             POCL_DEVICEn_PARAMETERS env. */
          if (snprintf (env_name, 1024, "POCL_%s%d_PARAMETERS", dev_name, j) < 0)
            POCL_ABORT("Unable to generate the env string.");

          pocl_devices[dev_index].ops->init(&pocl_devices[dev_index], getenv(env_name));

          if (dev_index == 0)
            pocl_devices[dev_index].type |= CL_DEVICE_TYPE_DEFAULT;
          
          ++dev_index;
        }
    }

  init_done = 1;
  POCL_UNLOCK(pocl_init_lock);
}
