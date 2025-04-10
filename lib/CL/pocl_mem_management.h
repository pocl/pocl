/* pocl_mem_management.c - manage allocation of runtime objects

   Copyright (c) 2014 Ville Korhonen
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

#include "pocl_cl.h"
#include "utlist.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef USE_POCL_MEMMANAGER

void pocl_init_mem_manager (void);

cl_event pocl_mem_manager_new_event (void);

void pocl_mem_manager_free_event (cl_event event);

_cl_command_node* pocl_mem_manager_new_command (void);

void pocl_mem_manager_free_command (_cl_command_node *cmd_ptr);

event_node* pocl_mem_manager_new_event_node ();

void pocl_mem_manager_free_event_node (event_node *ed);

#else

#define pocl_init_mem_manager() NULL

cl_event pocl_mem_manager_new_event ();

#define pocl_mem_manager_free_event(event) POCL_MEM_FREE(event)

#define pocl_mem_manager_new_command() \
  (_cl_command_node*) calloc (1, sizeof (_cl_command_node))

/* TODO: Should we free the migr_infos here or in the event? For user events
   there are no commands. */
#define pocl_mem_manager_free_command(cmd)                                    \
  if ((cmd))                                                                  \
    {                                                                         \
      if ((cmd)->buffered)                                                    \
        {                                                                     \
          POCL_MEM_FREE ((cmd)->sync.syncpoint.sync_point_wait_list);         \
        }                                                                     \
      pocl_buffer_migration_info *mi, *tmp;                                   \
      LL_FOREACH_SAFE ((cmd)->migr_infos, mi, tmp)                            \
        {                                                                     \
          POname (clReleaseMemObject (mi->buffer));                           \
          POCL_MEM_FREE (mi);                                                 \
        }                                                                     \
    }                                                                         \
  POCL_MEM_FREE ((cmd));

#define pocl_mem_manager_new_event_node() \
  (event_node*) calloc (1, sizeof (event_node))

#define pocl_mem_manager_free_event_node(en) POCL_MEM_FREE(en)

pocl_buffer_migration_info *pocl_append_unique_migration_info (
  pocl_buffer_migration_info *list, cl_mem buffer, char read_only);

pocl_buffer_migration_info *
pocl_deep_copy_migration_info_list (pocl_buffer_migration_info *list,
                                    int retain);

int pocl_create_migration_commands (cl_device_id dev,
                                    cl_event *ev_export_p,
                                    cl_event user_cmd,
                                    cl_mem mem,
                                    pocl_mem_identifier *gmem,
                                    const char readonly,
                                    cl_command_type command_type,
                                    cl_mem_migration_flags mig_flags,
                                    uint64_t migration_size,
                                    cl_event *prev_migr_event);

pocl_buffer_migration_info *
pocl_convert_to_subbuffer_migrations (pocl_buffer_migration_info *buffer_usage,
                                      cl_int *err);

/* Sets the indirect_raw_ptrs of the kernel to the given list array
 * of pointers.
 *
 * Clears up the possible previously set pointers.
 */
void
pocl_reset_indirect_ptrs (cl_kernel kernel, void **ptrs, size_t n);

/* Increments a buffer's reference counter. */
#define POCL_RETAIN_BUFFER_UNLOCKED(__OBJ__)                                  \
  do                                                                          \
    {                                                                         \
      ++((__OBJ__)->pocl_refcount);                                           \
    }                                                                         \
  while (0)

#endif

/**
 * Get the device memory pointer of the supplied pocl argument.
 *
 * \param global_mem_id [in] This is needed to get the device specific pointer.
 * \return NULL if arg->value is NULL and otherwise the requested pointer.
 */
POCL_EXPORT void *
pocl_cpu_get_ptr (struct pocl_argument *arg, unsigned global_mem_id);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif
