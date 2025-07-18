/* Definition of available OpenCL devices.

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2018 Pekka Jääskeläinen
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#define _GNU_SOURCE

#include <string.h>

#ifdef __linux__
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <ucontext.h>
#endif

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "devices.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_dynlib.h"
#include "pocl_export.h"
#include "pocl_runtime_config.h"
#include "pocl_tracing.h"
#include "pocl_util.h"
#include "pocl_version.h"

#include "utlist.h"
#include "utlist_addon.h"

#ifdef ENABLE_RDMA
#include "pocl_rdma.h"
#endif

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

#ifdef BUILD_BASIC
#include "basic/basic.h"
#endif
#ifdef BUILD_PTHREAD
#include "pthread/pocl-pthread.h"
#endif
#ifdef BUILD_TBB
#include "tbb/tbb.h"
#endif

#ifdef TCE_AVAILABLE
#include "tce/ttasim/ttasim.h"
#endif

#ifdef BUILD_HSA
#include "hsa/pocl-hsa.h"
#endif

#ifdef BUILD_CUDA
#include "cuda/pocl-cuda.h"
#endif

#if defined(BUILD_ALMAIF)
#include "almaif/almaif.h"
#endif

#ifdef BUILD_PROXY
#include "proxy/pocl_proxy.hpp"
#endif

#ifdef BUILD_VULKAN
#include "vulkan/pocl-vulkan.h"
#endif

#ifdef BUILD_LEVEL0
#include "level0/pocl-level0.h"
#endif

#define MAX_ENV_NAME_LEN 1024

#ifdef BUILD_REMOTE_CLIENT
#include "remote/remote.h"
#endif

#define MAX_DEV_NAME_LEN 64

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* the enabled devices */
/*
  IMPORTANT: utlist_addon.h macros are used to atomically access
  pocl_devices' next pointers. pocl_devices should only be accessed
  through these macros. LL_DELETE shouldn't be used as a devices' address
  must remain valid. LL_PREPEND shouldn't be used as well as it modifies the
  head.
*/
/* Head for the pocl_devices linked list*/
struct _cl_device_id *pocl_devices = NULL;
uint64_t pocl_num_devices = 0;

#ifdef ENABLE_LOADABLE_DRIVERS
#define INIT_DEV(ARG) NULL
#else
#define INIT_DEV(ARG) pocl_##ARG##_init_device_ops
#endif

const char *
pocl_get_device_name (unsigned index)
{

  if (index < POCL_ATOMIC_LOAD (pocl_num_devices))
    {
      cl_device_id device;
      unsigned i = 0;
      LL_FOREACH_ATOMIC (pocl_devices, device)
      {
        if (i == index)
          return device->long_name;
        i++;
        }
    }
  return NULL;
}

/* Init function prototype */
typedef void (*init_device_ops)(struct pocl_device_ops*);

/* All init function for device operations available to pocl */
static init_device_ops pocl_devices_init_ops[] = {
#ifdef BUILD_BASIC
  INIT_DEV (basic),
#endif
#ifdef BUILD_PTHREAD
  INIT_DEV (pthread),
#endif
#ifdef TCE_AVAILABLE
  INIT_DEV (ttasim),
#endif
#ifdef BUILD_TBB
  INIT_DEV (tbb),
#endif
#ifdef BUILD_HSA
  INIT_DEV (hsa),
#endif
#ifdef BUILD_CUDA
  INIT_DEV (cuda),
#endif
#ifdef BUILD_ALMAIF
  INIT_DEV (almaif),
#endif
#ifdef BUILD_PROXY
  INIT_DEV (proxy),
#endif
#ifdef BUILD_VULKAN
  INIT_DEV (vulkan),
#endif
#ifdef BUILD_LEVEL0
  INIT_DEV (level0),
#endif
#ifdef BUILD_REMOTE_CLIENT
  INIT_DEV (remote),
#endif
};

#define POCL_NUM_DEVICE_TYPES (sizeof(pocl_devices_init_ops) / sizeof((pocl_devices_init_ops)[0]))

char pocl_device_types[POCL_NUM_DEVICE_TYPES][33] = {
#ifdef BUILD_BASIC
  "basic",
#endif
#ifdef BUILD_PTHREAD
  "pthread",
#endif
#ifdef BUILD_TBB
  "tbb",
#endif
#ifdef TCE_AVAILABLE
  "ttasim",
#endif
#ifdef BUILD_HSA
  "hsa",
#endif
#ifdef BUILD_CUDA
  "cuda",
#endif
#ifdef BUILD_ALMAIF
  "almaif",
#endif
#ifdef BUILD_PROXY
  "proxy",
#endif
#ifdef BUILD_VULKAN
  "vulkan",
#endif
#ifdef BUILD_LEVEL0
  "level0",
#endif
#ifdef BUILD_REMOTE_CLIENT
  "remote",
#endif
};

static struct pocl_device_ops pocl_device_ops[POCL_NUM_DEVICE_TYPES];

extern pocl_lock_t pocl_context_handling_lock;

POCL_EXPORT int pocl_offline_compile = 0;

/* first setup */
static unsigned first_init_done = 0;
static unsigned init_in_progress = 0;
static uint64_t device_count[POCL_NUM_DEVICE_TYPES];
/* Indexes each device added to the platform by setting the device id. First
 * used and modified during init, to index devices present since launch. May
 * also used and modified when devices are dynamically added. */
static uint64_t dev_index;

/* after calling drivers uninit, we may have to re-init the devices. */
static unsigned devices_active = 0;

extern pocl_lock_t pocl_init_lock;

#ifdef ENABLE_LOADABLE_DRIVERS

static void *pocl_device_handles[POCL_NUM_DEVICE_TYPES];

static void
get_pocl_device_lib_path (char *result, char *device_name, int absolute_path)
{
  const char *soname = NULL;
  if (absolute_path
      && (soname = pocl_dynlib_pathname ((void *)get_pocl_device_lib_path)))
    {
      strcpy (result, soname);
      char *last_slash = strrchr (result, POCL_PATH_SEPARATOR[0]);
      *(++last_slash) = '\0';
      if (strlen (result) > 0)
        {
#ifdef ENABLE_POCL_BUILDING
          if (pocl_get_bool_option ("POCL_BUILDING", 0))
            {
              strcat (result, "devices");
              strcat (result, POCL_PATH_SEPARATOR);
              if (strncmp(device_name, "ttasim", 6) == 0)
                {
                  strcat (result, "tce");
                }
              else
                {
                  strcat (result, device_name);
                }
            }
          else
#endif
            {
              strcat (result, POCL_INSTALL_FROM_LIB_TO_PRIVATE_LIBDIR);
            }
          strcat (result, POCL_PATH_SEPARATOR);
#ifdef _WIN32
          strcat (result, "pocl-devices-");
#else
          strcat (result, "libpocl-devices-");
#endif
          strcat (result, device_name);
#if defined(_WIN32)
          strcat (result, ".dll");
#elif defined(__APPLE__)
          strcat (result, ".dylib");
#else
          strcat (result, ".so");
#endif
        }
    }
  else
    {
      strcat (result, POCL_INSTALL_FROM_LIB_TO_PRIVATE_LIBDIR);
      strcat (result, POCL_PATH_SEPARATOR);
#ifdef _WIN32
      strcat (result, "pocl-devices-");
#else
      strcat (result, "libpocl-devices-");
#endif
      strcat (result, device_name);
#if defined(_WIN32)
      strcat (result, ".dll");
#elif defined(__APPLE__)
      strcat (result, ".dylib");
#else
      strcat (result, ".so");
#endif
    }
}
#endif

/**
 * Get the number of specified devices from environment
 */
int pocl_device_get_env_count(const char *dev_type)
{
  const char *dev_env = getenv(POCL_DEVICES_ENV);
  char *ptr, *saveptr = NULL, *tofree, *token;
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

unsigned int
pocl_get_devices (cl_device_type device_type, cl_device_id *devices,
                  unsigned int num_devices)
{
  unsigned int dev_added = 0;
  cl_device_id device;

  LL_FOREACH_ATOMIC (pocl_devices, device)
  {
    if (!pocl_offline_compile && (*device->available == CL_FALSE))
      continue;

    if (device_type == CL_DEVICE_TYPE_DEFAULT)
      {
        /* according to spec, DEFAULT query must not return
         * a custom device UNLESS it's the only device */
        if (device->type == CL_DEVICE_TYPE_CUSTOM && device->next)
          continue;

        devices[dev_added] = device;
        ++dev_added;
        break;
      }

    if (device->type & device_type)
      {
        if (dev_added < num_devices)
          {
            devices[dev_added] = device;
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
  unsigned int count = 0;
  cl_device_id device;

  if (device_type == CL_DEVICE_TYPE_DEFAULT)
    {
      return pocl_devices ? 1 : 0;
    }

  LL_FOREACH_ATOMIC (pocl_devices, device)
  {
    if (!pocl_offline_compile && (*device->available == CL_FALSE))
      continue;

    if (device->type & device_type)
      {
        ++count;
      }
    }

  return count;
}

cl_int
pocl_uninit_devices ()
{
  cl_int retval = CL_SUCCESS;

  POCL_LOCK (pocl_init_lock);
  if ((!devices_active) || (POCL_ATOMIC_LOAD (pocl_num_devices) == 0))
    goto FINISH;

  POCL_MSG_PRINT_GENERAL ("UNINIT all devices\n");

  pocl_event_tracing_finish ();
  pocl_async_callback_finish ();

  unsigned i, j;
  cl_device_id device = pocl_devices;

  cl_device_id d;
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      if (pocl_devices_init_ops[i] == NULL)
        continue;
      assert (pocl_device_ops[i].init);

      j = 0;
      LL_FOREACH_ATOMIC (device, device)
      {
        d = device;
        if (*(d->available) == CL_FALSE)
          continue;
        if (d->ops->reinit == NULL || d->ops->uninit == NULL)
          continue;
        cl_int ret = d->ops->uninit (j, d);
        if (ret != CL_SUCCESS)
          {
            retval = ret;
            goto FINISH;
          }
#ifdef ENABLE_LOADABLE_DRIVERS
          if (pocl_device_handles[i] != NULL)
          pocl_dynlib_close (pocl_device_handles[i]);
#endif
          j++;
        }
    }

FINISH:
#ifdef ENABLE_SIGFPE_HANDLER
  pocl_destroy_sigfpe_handler ();
#endif

  devices_active = 0;
  POCL_UNLOCK (pocl_init_lock);

  return retval;
}

static cl_int
pocl_reinit_devices ()
{
  assert (first_init_done);
  cl_int retval = CL_SUCCESS;

  if (devices_active)
    return retval;

  if (POCL_ATOMIC_LOAD (pocl_num_devices) == 0)
    return CL_DEVICE_NOT_FOUND;

  POCL_MSG_WARN ("REINIT all devices\n");

  /*  TODO: reinit tracing */
  /* pocl_event_tracing_init (); */
  pocl_async_callback_init ();

  unsigned i, j;
  cl_device_id device = pocl_devices;

  char env_name[1024];
  char dev_name[MAX_DEV_NAME_LEN] = { 0 };
  cl_device_id d;
  /* Init infos for each probed devices */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      pocl_str_toupper (dev_name, pocl_device_ops[i].device_name);
      assert (pocl_device_ops[i].init);

      j = 0;
      LL_FOREACH_ATOMIC (device, device)
      {
        d = device;
        if (*(d->available) == CL_FALSE)
          continue;
        if (d->ops->reinit == NULL || d->ops->uninit == NULL)
          continue;
        snprintf (env_name, 1024, "POCL_%s%d_PARAMETERS", dev_name, j);
        cl_int ret = d->ops->reinit (j, d, getenv (env_name));
        if (ret != CL_SUCCESS)
          {
            retval = ret;
            goto FINISH;
          }

        j++;
        }
    }

FINISH:

  devices_active = 1;
  return retval;
}

/**
 * Callback function that dynamically adds and initializes devices. It is
 * registered by pocl_init_device_discovery(), for device drivers that support
 * device discovery. Can be called by any and more than one device drivers.
 */
cl_int
add_discovered_device_callback (const char *dev_parameters,
                                unsigned pocl_dev_type_idx,
                                cl_platform_id pocl_dev_platform)
{
  assert (first_init_done);

  POCL_LOCK (pocl_init_lock);
  int errcode = CL_SUCCESS;

  char dev_name[MAX_DEV_NAME_LEN] = { 0 };
  pocl_str_toupper (dev_name, pocl_device_ops[pocl_dev_type_idx].device_name);
  assert (pocl_device_ops[pocl_dev_type_idx].init);

#ifdef POCL_DEBUG_MESSAGES
  const char *debug = pocl_get_string_option ("POCL_DEBUG", "0");
  pocl_debug_messages_setup (debug);
  pocl_stderr_is_a_tty = isatty (fileno (stderr));
#endif

  cl_device_id dev;
  dev = (cl_device_id)calloc (1, sizeof (*dev));

  dev->ops = &pocl_device_ops[pocl_dev_type_idx];
  dev->dev_id = dev_index;
  dev->global_mem_id = dev_index;
  POCL_INIT_OBJECT (dev, pocl_dev_platform);
  dev->driver_version = POCL_VERSION_FULL;
  if (dev->version == NULL)
    dev->version = "OpenCL 3.0 pocl";

  errcode = dev->ops->init (pocl_dev_type_idx, dev, dev_parameters);
  POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), errcode,
                      "Device %i / %s initialization failed! \n",
                      pocl_dev_type_idx, dev_name);

  LL_APPEND_ATOMIC (pocl_devices, dev);
  POCL_ATOMIC_INC (device_count[pocl_dev_type_idx]);
  POCL_ATOMIC_INC (pocl_num_devices);
  POCL_ATOMIC_INC (dev_index);
  devices_active = 1;

  POCL_UNLOCK (pocl_init_lock);
  return errcode;

ERROR:
  POCL_MEM_FREE (dev);
  POCL_UNLOCK (pocl_init_lock);
  return errcode;
}

/**
 * Initialize discovery in device driver
 */
cl_int
pocl_init_device_discovery (cl_platform_id platform)
{
  assert (first_init_done);
  /* Discovery initialization should happen once and only by one thread. */
  static uint64_t first_disco_init_done = 0;
  if (POCL_ATOMIC_CAS (&first_disco_init_done, 0, 1))
    return CL_SUCCESS;

  int errcode = CL_SUCCESS;

#ifdef POCL_DEBUG_MESSAGES
  const char *debug = pocl_get_string_option ("POCL_DEBUG", "0");
  pocl_debug_messages_setup (debug);
  pocl_stderr_is_a_tty = isatty (fileno (stderr));
#endif

  /* If device discovery is disabled by the env variable then return
   * CL_SUCCESS.
   */
  if (!pocl_get_bool_option (POCL_DISCOVERY_ENV, 0))
    goto ERROR;

  for (unsigned i = 0; i < POCL_NUM_DEVICE_TYPES; i++)
    {
      if (pocl_device_ops[i].init_discovery == NULL)
        continue;

      errcode = pocl_device_ops[i].init_discovery (
        &add_discovered_device_callback, i, platform);
      POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), errcode,
                          "Discovery initialization failed for device: %s \n",
                          pocl_device_ops[i].device_name);
    }

ERROR:
  return errcode;
}

cl_int
pocl_init_devices (cl_platform_id platform)
{
  int errcode = CL_SUCCESS;

  /* This is a workaround to a nasty problem with libhwloc: When
     initializing basic, it calls libhwloc to query device info.
     In case libhwloc has the OpenCL plugin installed, it initializes
     it and it leads to initializing pocl again which leads to an
     infinite loop. This only protects against recursive calls of
     pocl_init_devices(), so must be done without pocl_init_lock held. */
  if (init_in_progress)
    return CL_SUCCESS; /* debatable, but what else can we do ? */

  POCL_LOCK (pocl_init_lock);
  init_in_progress = 1;

  if (first_init_done)
    {
      if (!devices_active)
        {
          POCL_MSG_PRINT_GENERAL ("FIRST INIT done; REINIT all devices\n");
          pocl_reinit_devices (); // TODO err check
        }
      errcode = pocl_num_devices ? CL_SUCCESS : CL_DEVICE_NOT_FOUND;
      goto ERROR;
    }
  else
    {
      POCL_INIT_LOCK (pocl_context_handling_lock);
    }

  /* first time initialization */
  unsigned i, j;
  char env_name[MAX_ENV_NAME_LEN] = { 0 };
  char dev_name[MAX_DEV_NAME_LEN] = { 0 };

  /* Set a global debug flag, so we don't have to call pocl_get_bool_option
   * everytime we use the debug macros */
#ifdef POCL_DEBUG_MESSAGES
  const char* debug = pocl_get_string_option ("POCL_DEBUG", "0");
  pocl_debug_messages_setup (debug);
#endif

  POCL_GOTO_ERROR_ON ((pocl_cache_init_topdir ()), CL_DEVICE_NOT_FOUND,
                      "Cache directory initialization failed");

  pocl_init_builtin_kernel_metadata ();

  pocl_event_tracing_init ();

  pocl_async_callback_init ();

#ifdef HAVE_SLEEP
  int delay = pocl_get_int_option ("POCL_STARTUP_DELAY", 0);
  if (delay > 0)
    sleep (delay);
#endif

#ifdef ENABLE_SIGFPE_HANDLER
  pocl_install_sigfpe_handler ();
#endif
#ifdef ENABLE_SIGUSR2_HANDLER
  if (pocl_get_bool_option ("POCL_SIGUSR2_HANDLER", 0))
    {
      pocl_install_sigusr2_handler ();
    }
#endif

  pocl_offline_compile = pocl_get_bool_option ("POCL_OFFLINE_COMPILE", 0);

  /* Init operations */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
#ifdef ENABLE_LOADABLE_DRIVERS
      if (pocl_devices_init_ops[i] == NULL)
        {
          char device_library[PATH_MAX] = "";
          char init_device_ops_name[MAX_DEV_NAME_LEN + 21] = "";
          get_pocl_device_lib_path (device_library, pocl_device_types[i], 1);
          pocl_device_handles[i] = pocl_dynlib_open (device_library, 1, 0);
          if (pocl_device_handles[i] == NULL)
            {
              POCL_MSG_WARN ("Loading %s failed.\n", device_library);

              /* Try again with just the *.so filename */
              device_library[0] = 0;
              get_pocl_device_lib_path (device_library,
                                        pocl_device_types[i], 0);
              pocl_device_handles[i] = pocl_dynlib_open (device_library, 1, 0);
              if (pocl_device_handles[i] == NULL)
                {
                  POCL_MSG_WARN ("Loading %s failed\n", device_library);
                  device_count[i] = 0;
                  continue;
                }
              else
                {
                  POCL_MSG_WARN ("Fallback loading %s succeeded\n",
                                 device_library);
                }
            }
          strcat (init_device_ops_name, "pocl_");
          strcat (init_device_ops_name, pocl_device_types[i]);
          strcat (init_device_ops_name, "_init_device_ops");
          pocl_devices_init_ops[i]
            = (init_device_ops)pocl_dynlib_symbol_address (
              pocl_device_handles[i], init_device_ops_name);
          if (pocl_devices_init_ops[i] == NULL)
            {
              POCL_MSG_ERR ("Loading symbol %s from %s failed\n",
                            init_device_ops_name, device_library);
              device_count[i] = 0;
              continue;
            }
        }
      else
        {
          pocl_device_handles[i] = NULL;
        }
#else
      assert (pocl_devices_init_ops[i] != NULL);
#endif
      pocl_devices_init_ops[i](&pocl_device_ops[i]);
      assert(pocl_device_ops[i].device_name != NULL);

      /* Probe and add the result to the number of probed devices */
      assert(pocl_device_ops[i].probe);
      device_count[i] = pocl_device_ops[i].probe(&pocl_device_ops[i]);
      pocl_num_devices += device_count[i];
    }

  dev_index = 0;

  const char *dev_env = pocl_get_string_option (POCL_DEVICES_ENV, NULL);

  /* If device discovery is enabled and no devices are found through probing
   * then, we return with CL_SUCCESS after which discovery process is
   * initialised. The disocvery process may find devices, hence we don't return
   * with error. */
  if (pocl_get_bool_option (POCL_DISCOVERY_ENV, 0) && 0 == pocl_num_devices)
    {
      POCL_MSG_WARN (
        "No devices found by probing: %s=%s. Trying through discovery. \n",
        POCL_DEVICES_ENV, dev_env);
      first_init_done = 1;
      errcode = pocl_init_device_discovery (platform);
      goto ERROR;
    }
  POCL_GOTO_ERROR_ON ((pocl_num_devices == 0), CL_DEVICE_NOT_FOUND,
                      "no devices found by probing. %s=%s\n", POCL_DEVICES_ENV,
                      dev_env);

  /* Init infos for each probed devices */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      if (pocl_devices_init_ops[i] == NULL)
        continue;
      pocl_str_toupper (dev_name, pocl_device_ops[i].device_name);
      assert(pocl_device_ops[i].init);

      for (j = 0; j < device_count[i]; ++j)
        {
          cl_device_id dev;
          dev = (cl_device_id)calloc (1, sizeof (*dev));

          dev->ops = &pocl_device_ops[i];
          dev->dev_id = dev_index;
          /* The default value for the global memory space identifier is
             the same as the device id. The device instance can then override
             it to point to some other device's global memory id in case of
             a shared global memory. */
          dev->global_mem_id = dev_index;
          POCL_INIT_OBJECT (dev, platform);
          dev->driver_version = pocl_get_string_option (
              "POCL_DRIVER_VERSION_OVERRIDE", POCL_VERSION_FULL);

          if (dev->version == NULL)
            dev->version = "OpenCL 3.0 pocl";

          /* Check if there are device-specific parameters set in the
             POCL_DEVICEn_PARAMETERS env. */
          POCL_GOTO_ERROR_ON (
              (snprintf (env_name, MAX_ENV_NAME_LEN,
                         "POCL_%s%d_PARAMETERS", dev_name, j)
               < 0),
              CL_OUT_OF_HOST_MEMORY, "Unable to generate the env string.");

          errcode = dev->ops->init (j, dev, getenv (env_name));
          POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), errcode,
                              "Device %i / %s initialization failed! \n", j,
                              dev_name);

          LL_APPEND_ATOMIC (pocl_devices, dev);

          ++dev_index;
        }
      if (pocl_device_ops[i].post_init != NULL)
        {
          pocl_device_ops[i].post_init(&pocl_device_ops[i]);
        }
    }
  first_init_done = 1;
  devices_active = 1;
  errcode = pocl_init_device_discovery (platform);
ERROR:
  init_in_progress = 0;
  POCL_UNLOCK (pocl_init_lock);
  return errcode;
}

#ifdef BUILD_ICD
void
pocl_set_devices_dispatch_data (void *disp_data)
{
  cl_device_id device;

  LL_FOREACH_ATOMIC (pocl_devices, device) { device->disp_data = disp_data; }
}
#endif
