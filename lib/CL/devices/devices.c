/* Definition of available OpenCL devices.

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2018 Pekka Jääskeläinen

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

#define _GNU_SOURCE

#include <string.h>
#include <ctype.h>

#ifdef __linux__
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <ucontext.h>
#endif

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "common.h"
#include "config.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_runtime_config.h"
#include "pocl_tracing.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#if defined(TCE_AVAILABLE)
#include "tce/ttasim/ttasim.h"
#endif

#include "hsa/pocl-hsa.h"

#if defined(BUILD_ACCEL)
#include "accel/accel.h"
#endif

#define MAX_DEV_NAME_LEN 64

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef HAVE_LIBDL
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#include <dlfcn.h>
#endif

/* the enabled devices */
static struct _cl_device_id* pocl_devices = NULL;
unsigned int pocl_num_devices = 0;

/* Init function prototype */
typedef void (*init_device_ops)(struct pocl_device_ops*);

/* All init function for device operations available to pocl */
static init_device_ops pocl_devices_init_ops[] = {
#ifdef BUILD_BASIC
  NULL,
#endif
#ifdef BUILD_PTHREAD
  NULL,
#endif
#if defined(TCE_AVAILABLE)
  NULL,
#endif
#if defined(BUILD_HSA)
  NULL,
#endif
#if defined(BUILD_CUDA)
  NULL,
#endif
#if defined(BUILD_ACCEL)
  NULL,
#endif
};

#define POCL_NUM_DEVICE_TYPES (sizeof(pocl_devices_init_ops) / sizeof((pocl_devices_init_ops)[0]))

char pocl_device_types[POCL_NUM_DEVICE_TYPES][30] = {
#ifdef BUILD_BASIC
  "basic",
#endif
#ifdef BUILD_PTHREAD
  "pthread",
#endif
#if defined(TCE_AVAILABLE)
  "ttasim",
#endif
#if defined(BUILD_HSA)
  "hsa",
#endif
#if defined(BUILD_CUDA)
  "cuda",
#endif
#if defined(BUILD_ACCEL)
  "accel",
#endif
};

static struct pocl_device_ops pocl_device_ops[POCL_NUM_DEVICE_TYPES];

// first setup
static unsigned first_init_done = 0;
static unsigned init_in_progress = 0;
static unsigned device_count[POCL_NUM_DEVICE_TYPES];

// after calling drivers uninit, we may have to re-init the devices.
static unsigned devices_active = 0;

static pocl_lock_t pocl_init_lock = POCL_LOCK_INITIALIZER;

static void *pocl_device_handles[POCL_NUM_DEVICE_TYPES];

#ifndef _MSC_VER
#define POCL_PATH_SEPARATOR "/"
#else
#define POCL_PATH_SEPARATOR "\\"
#endif

static void
get_pocl_device_lib_path (char *result, char *device_name)
{
  Dl_info info;
  if (dladdr ((void *)get_pocl_device_lib_path, &info))
    {
      char const *soname = info.dli_fname;
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
              strcat (result, POCL_PATH_SEPARATOR);
            }
          else
#endif
            {
              strcat (result, POCL_INSTALL_PRIVATE_LIBDIR_REL);
            }
          strcat (result, POCL_PATH_SEPARATOR);
          strcat (result, "libpocl-devices-");
          strcat (result, device_name);
          strcat (result, ".so");
          return;
        }
    }
}

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
pocl_get_devices(cl_device_type device_type, struct _cl_device_id **devices, unsigned int num_devices)
{
  unsigned int i, dev_added = 0;

  int offline_compile = pocl_get_bool_option("POCL_OFFLINE_COMPILE", 0);

  for (i = 0; i < pocl_num_devices; ++i)
    {
      if (!offline_compile && (pocl_devices[i].available != CL_TRUE))
        continue;

      if (device_type == CL_DEVICE_TYPE_DEFAULT)
        {
          devices[dev_added] = &pocl_devices[i];
          ++dev_added;
          break;
        }

      if (pocl_devices[i].type & device_type)
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

  int offline_compile = pocl_get_bool_option("POCL_OFFLINE_COMPILE", 0);

  for (i = 0; i < pocl_num_devices; ++i)
    {
      if (!offline_compile && (pocl_devices[i].available != CL_TRUE))
        continue;

      if (device_type == CL_DEVICE_TYPE_DEFAULT)
        return 1;

      if (pocl_devices[i].type & device_type)
        {
           ++count;
        }
    }

  return count;
}


static inline void
str_toupper(char *out, const char *in)
{
  int i;

  for (i = 0; in[i] != '\0'; i++)
    out[i] = toupper(in[i]);
  out[i] = '\0';
}

/* This ugly hack is required because:
 *
 * OpenCL 1.2 specification, 6.3 Operators :
 *
 * A divide by zero with integer types does not cause an exception
 * but will result in an unspecified value. Division by zero for
 * floating-point types will result in ±infinity or NaN as
 * prescribed by the IEEE-754 standard.
 *
 * FPU exceptions are masked by default on x86 linux, but integer divide
 * is not and there doesn't seem any sane way to mask it.
 *
 * This *might* be possible to fix with a LLVM pass (either check divisor
 * for 0, or perhaps some vector extension has a suitable instruction), but
 * it's likely to ruin the performance.
 */

#ifdef ENABLE_HOST_CPU_DEVICES
#ifdef __linux__
#ifdef __x86_64__

#define DIV_OPCODE_SIZE 1
#define DIV_OPCODE_MASK 0xf6

/* F6 /6, F6 /7, F7 /6, F7 /7 */
#define DIV_OPCODE_1 0xf6
#define DIV_OPCODE_2 0xf7
#define DIV_MODRM_OPCODE_EXT_1 0x38 //  /7
#define DIV_MODRM_OPCODE_EXT_2 0x30 //  /6

#define MODRM_SIZE 1
#define MODRM_MASK 0xC0
#define REG2_MASK 0x38
#define REG1_MASK 0x07
#define ADDR_MODE_INDIRECT_ONE_BYTE_OFFSET 0x40
#define ADDR_MODE_INDIRECT_FOUR_BYTE_OFFSET 0x80
#define ADDR_MODE_INDIRECT 0x0
#define ADDR_MODE_REGISTER_ONLY 0xC0
#define REG_SP 0x4
#define REG_BP 0x5
#define SIB_BYTE 1
#define IP_RELATIVE_INDEXING 4

static struct sigaction sigfpe_action, old_sigfpe_action;

static void
sigfpe_signal_handler (int signo, siginfo_t *si, void *data)
{
  ucontext_t *uc;
  uc = (ucontext_t *)data;
  unsigned char *eip = (unsigned char *)(uc->uc_mcontext.gregs[REG_RIP]);

  if ((signo == SIGFPE)
      && ((si->si_code == FPE_INTDIV) || (si->si_code == FPE_INTOVF)))
    {
      /* Luckily for us, div-by-0 exceptions do NOT advance the IP register,
       * so we have to disassemble the instruction (to know its length)
       * and move IP past it. */
      unsigned n = 0;

      /* skip all prefixes */
      while ((n < 4) && ((eip[n] & DIV_OPCODE_MASK) != DIV_OPCODE_MASK))
        ++n;

      /* too much prefixes = decoding failed */
      if (n >= 4)
        goto ORIGINAL_HANDLER;

      /* check opcode */
      unsigned opcode = eip[n];
      if ((opcode != DIV_OPCODE_1) && (opcode != DIV_OPCODE_2))
        goto ORIGINAL_HANDLER;
      n += DIV_OPCODE_SIZE;

      unsigned modrm = eip[n];
      unsigned modmask = modrm & MODRM_MASK;
      unsigned reg1mask = modrm & REG1_MASK;
      unsigned reg2mask = modrm & REG2_MASK;
      /* check opcode extension in ModR/M reg2 */
      if ((reg2mask != DIV_MODRM_OPCODE_EXT_1)
          && (reg2mask != DIV_MODRM_OPCODE_EXT_2))
        goto ORIGINAL_HANDLER;
      n += MODRM_SIZE;

      /* handle immediates/registers */
      if (modmask == ADDR_MODE_INDIRECT_ONE_BYTE_OFFSET)
        n += 1;
      if (modmask == ADDR_MODE_INDIRECT_FOUR_BYTE_OFFSET)
        n += 4;
      if (modmask == ADDR_MODE_INDIRECT)
        n += 0;
      if (modmask != ADDR_MODE_REGISTER_ONLY)
        {
          if (reg1mask == REG_SP)
            n += SIB_BYTE;
          if (reg1mask == REG_BP)
            n += IP_RELATIVE_INDEXING;
        }

      uc->uc_mcontext.gregs[REG_RIP] += n;
      return;
    }
  else
    {
    ORIGINAL_HANDLER:
      (*old_sigfpe_action.sa_sigaction) (signo, si, data);
    }
}

#endif
#endif
#endif

cl_int
pocl_uninit_devices ()
{
  cl_int retval = CL_SUCCESS;

  POCL_LOCK (pocl_init_lock);
  if ((!devices_active) || (pocl_num_devices == 0))
    goto FINISH;

  POCL_MSG_PRINT_GENERAL ("UNINIT all devices\n");

  unsigned i, j, dev_index;

  dev_index = 0;
  cl_device_id d;
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      if (pocl_devices_init_ops[i] == NULL)
        continue;
      assert (pocl_device_ops[i].init);
      for (j = 0; j < device_count[i]; ++j)
        {
          d = &pocl_devices[dev_index];
          if (d->available == 0)
            continue;
          if (d->ops->reinit == NULL || d->ops->uninit == NULL)
            continue;
          cl_int ret = d->ops->uninit (j, d);
          if (ret != CL_SUCCESS)
            {
              retval = ret;
              goto FINISH;
            }
          if (pocl_device_handles[i] != NULL)
            {
              dlclose (pocl_device_handles[i]);
            }
          ++dev_index;
        }
    }

FINISH:
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

  if (pocl_num_devices == 0)
    return CL_DEVICE_NOT_FOUND;

  POCL_MSG_WARN ("REINIT all devices\n");

  unsigned i, j, dev_index;

  dev_index = 0;
  cl_device_id d;
  /* Init infos for each probed devices */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      assert (pocl_device_ops[i].init);
      for (j = 0; j < device_count[i]; ++j)
        {
          d = &pocl_devices[dev_index];
          if (d->available == 0)
            continue;
          if (d->ops->reinit == NULL || d->ops->uninit == NULL)
            continue;
          cl_int ret = d->ops->reinit (j, d);
          if (ret != CL_SUCCESS)
            {
              retval = ret;
              goto FINISH;
            }

          ++dev_index;
        }
    }

FINISH:

  devices_active = 1;
  return retval;
}

cl_int
pocl_init_devices ()
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

  /* first time initialization */
  unsigned i, j, dev_index;
  char env_name[1024];
  char dev_name[MAX_DEV_NAME_LEN] = { 0 };

  /* Set a global debug flag, so we don't have to call pocl_get_bool_option
   * everytime we use the debug macros */
#ifdef POCL_DEBUG_MESSAGES
  const char* debug = pocl_get_string_option ("POCL_DEBUG", "0");
  pocl_debug_messages_setup (debug);
  pocl_stderr_is_a_tty = isatty(fileno(stderr));
#endif

  POCL_GOTO_ERROR_ON ((pocl_cache_init_topdir ()), CL_DEVICE_NOT_FOUND,
                      "Cache directory initialization failed");

  pocl_tracing_init ();

#ifdef HAVE_SLEEP
  int delay = pocl_get_int_option ("POCL_STARTUP_DELAY", 0);
  if (delay > 0)
    sleep (delay);
#endif


#ifdef ENABLE_HOST_CPU_DEVICES
#ifdef __linux__
#ifdef __x86_64__

  if (pocl_get_bool_option ("POCL_SIGFPE_HANDLER", 1))
    {

#ifdef OCS_AVAILABLE
      /* This is required to force LLVM to register its signal
       * handlers, before pocl registers its own SIGFPE handler.
       * LLVM otherwise calls this via
       *    pocl_llvm_build_program ->
       *    clang::PrintPreprocessedAction ->
       *    CreateOutputFile -> RemoveFileOnSignal
       * Registering our handlers before LLVM creates its sigaltstack
       * leads to interesting crashes & bugs later.
       */
      char random_empty_file[POCL_FILENAME_LENGTH];
      pocl_cache_tempname (random_empty_file, NULL, NULL);
      pocl_llvm_remove_file_on_signal (random_empty_file);
#endif

      POCL_MSG_PRINT_GENERAL ("Installing SIGFPE handler...\n");
      sigfpe_action.sa_flags = SA_RESTART | SA_SIGINFO;
      sigfpe_action.sa_sigaction = sigfpe_signal_handler;
      int res = sigaction (SIGFPE, &sigfpe_action, &old_sigfpe_action);
      assert (res == 0);
    }

#endif
#endif
#endif

  /* Init operations */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      if (pocl_devices_init_ops[i] == NULL)
        {
          char device_library[PATH_MAX] = "";
          get_pocl_device_lib_path (device_library, pocl_device_types[i]);
          pocl_device_handles[i] = dlopen (device_library, RTLD_LAZY);
          char init_device_ops_name[MAX_DEV_NAME_LEN + 21] = "";
          strcat (init_device_ops_name, "pocl_");
          strcat (init_device_ops_name, pocl_device_types[i]);
          strcat (init_device_ops_name, "_init_device_ops");
          if (pocl_device_handles[i] != NULL)
            {
              pocl_devices_init_ops[i] = (init_device_ops)dlsym (
                  pocl_device_handles[i], init_device_ops_name);
              if (pocl_devices_init_ops[i] != NULL)
                {
                  pocl_devices_init_ops[i](&pocl_device_ops[i]);
                }
              else
                {
                  POCL_MSG_ERR ("Loading symbol %s from %s failed: %s\n",
                                init_device_ops_name, device_library,
                                dlerror ());
                  device_count[i] = 0;
                  continue;
                }
            }
          else
            {
              POCL_MSG_WARN ("Loading %s failed: %s\n", device_library,
                             dlerror ());
              device_count[i] = 0;
              continue;
            }
        }
      else
        {
          pocl_device_handles[i] = NULL;
        }
      pocl_devices_init_ops[i](&pocl_device_ops[i]);
      assert(pocl_device_ops[i].device_name != NULL);

      /* Probe and add the result to the number of probed devices */
      assert(pocl_device_ops[i].probe);
      device_count[i] = pocl_device_ops[i].probe(&pocl_device_ops[i]);
      pocl_num_devices += device_count[i];
    }

  const char *dev_env = pocl_get_string_option (POCL_DEVICES_ENV, NULL);
  POCL_GOTO_ERROR_ON ((pocl_num_devices == 0), CL_DEVICE_NOT_FOUND,
                      "no devices found. %s=%s\n", POCL_DEVICES_ENV, dev_env);

  pocl_devices = (struct _cl_device_id*) calloc(pocl_num_devices, sizeof(struct _cl_device_id));
  POCL_GOTO_ERROR_ON ((pocl_devices == NULL), CL_OUT_OF_HOST_MEMORY,
                      "Can not allocate memory for devices\n");

  dev_index = 0;
  /* Init infos for each probed devices */
  for (i = 0; i < POCL_NUM_DEVICE_TYPES; ++i)
    {
      if (pocl_devices_init_ops[i] == NULL)
        continue;
      str_toupper (dev_name, pocl_device_ops[i].device_name);
      assert(pocl_device_ops[i].init);
      for (j = 0; j < device_count[i]; ++j)
        {
          cl_device_id dev = &pocl_devices[dev_index];
          dev->ops = &pocl_device_ops[i];
          dev->dev_id = dev_index;
          POCL_INIT_OBJECT (dev);
          dev->driver_version = PACKAGE_VERSION;
          if (dev->version == NULL)
            dev->version = "OpenCL 2.0 pocl";
          dev->short_name = strdup (dev->ops->device_name);
          /* The default value for the global memory space identifier is
             the same as the device id. The device instance can then override
             it to point to some other device's global memory id in case of
             a shared global memory. */
          pocl_devices[dev_index].global_mem_id = dev_index;

          /* Check if there are device-specific parameters set in the
             POCL_DEVICEn_PARAMETERS env. */
          POCL_GOTO_ERROR_ON (
              (snprintf (env_name, 1024, "POCL_%s%d_PARAMETERS", dev_name, j)
               < 0),
              CL_OUT_OF_HOST_MEMORY, "Unable to generate the env string.");
          errcode = pocl_devices[dev_index].ops->init (
              j, &pocl_devices[dev_index], getenv (env_name));
          POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), CL_DEVICE_NOT_AVAILABLE,
                              "Device %i / %s initialization failed!", j,
                              dev_name);

          ++dev_index;
        }
    }

  first_init_done = 1;
  devices_active = 1;
ERROR:
  init_in_progress = 0;
  POCL_UNLOCK (pocl_init_lock);
  return errcode;
}

int pocl_get_unique_global_mem_id ()
{
  static int global_id_counter = 1;
  return global_id_counter++;
}
