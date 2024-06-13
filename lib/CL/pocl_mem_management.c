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

#include "pocl_mem_management.h"
#include "pocl.h"
#include "pocl_util.h"

#include "utlist.h"
#include <string.h>

#ifndef USE_POCL_MEMMANAGER

cl_event pocl_mem_manager_new_event ()
{
  cl_event ev = (cl_event) calloc (1, sizeof (struct _cl_event));
  if (ev != NULL)
    POCL_INIT_OBJECT(ev);
  return ev;
}

#else

typedef struct _mem_manager
{
  pocl_lock_t event_lock;
  pocl_lock_t cmd_lock;
  pocl_lock_t event_node_lock;

  cl_event event_list;
  _cl_command_node *volatile cmd_list;
  event_node *event_node_list;
} pocl_mem_manager;


static pocl_mem_manager *mm = NULL;

void pocl_init_mem_manager (void)
{
  static unsigned int init_done = 0;
  static pocl_lock_t pocl_init_lock;

  if(!init_done)
    {
      POCL_INIT_LOCK(pocl_init_lock);
      init_done = 1;
    }
  POCL_LOCK(pocl_init_lock);
  if (!mm)
    {
      mm = (pocl_mem_manager*) calloc (1, sizeof (pocl_mem_manager));
      POCL_INIT_LOCK (mm->event_lock);
      POCL_INIT_LOCK (mm->cmd_lock);
      POCL_INIT_LOCK (mm->event_node_lock);
    }
  POCL_UNLOCK(pocl_init_lock);
}

cl_event pocl_mem_manager_new_event ()
{
  cl_event ev = NULL;
  POCL_LOCK (mm->event_lock);
  if ((ev = mm->event_list))
    {
      LL_DELETE (mm->event_list, ev);
      POCL_UNLOCK (mm->event_lock);
      POCL_INIT_OBJECT (ev); /* reinit the pocl_lock mutex */
      return ev;
    }
  POCL_UNLOCK (mm->event_lock);

  ev = (struct _cl_event*) calloc (1, sizeof (struct _cl_event));
  POCL_INIT_OBJECT(ev);
  return ev;
}

void pocl_mem_manager_free_event (cl_event event)
{
  assert (event->status <= CL_COMPLETE);
  POCL_LOCK (mm->event_lock);
  LL_PREPEND (mm->event_list, event);
  POCL_UNLOCK(mm->event_lock);
}

_cl_command_node* pocl_mem_manager_new_command ()
{
  _cl_command_node *cmd = NULL;
  POCL_LOCK (mm->cmd_lock);
  if ((cmd = mm->cmd_list))
    LL_DELETE (mm->cmd_list, cmd);
  POCL_UNLOCK (mm->cmd_lock);
  
  if (cmd)
    {
      memset (cmd, 0, sizeof (struct _cl_command_node));
      return cmd;
    }
  return (_cl_command_node*) calloc (1, sizeof (_cl_command_node));
}

void pocl_mem_manager_free_command (_cl_command_node *cmd_ptr)
{
  if (cmd && cmd->buffered)
    {
      /* TODO: recycle these somehow? */
      POCL_MEM_FREE (cmd->sync.syncpoint.sync_point_wait_list);
      POCL_MEM_FREE (cmd->memobj_list);
      POCL_MEM_FREE (cmd->readonly_flag_list);
    }
  POCL_LOCK (mm->cmd_lock);
  LL_PREPEND (mm->cmd_list, cmd_ptr);
  POCL_UNLOCK(mm->cmd_lock);
}

event_node* pocl_mem_manager_new_event_node ()
{
  event_node *ed = NULL;
  POCL_LOCK(mm->event_node_lock);
  if ((ed = mm->event_node_list))
    LL_DELETE (mm->event_node_list, ed);
  POCL_UNLOCK (mm->event_node_lock);
  
  if (ed)
    {
      memset (ed, 0, sizeof(event_node));
      return ed;
    }

  return calloc (1, sizeof (event_node));
}

void pocl_mem_manager_free_event_node (event_node *ed)
{
  POCL_LOCK (mm->event_node_lock);
  LL_PREPEND (mm->event_node_list, ed);
  POCL_UNLOCK (mm->event_node_lock);
}

#endif

/**
 * Set the sub-buffers of the parent buffer after a parent buffer update.
 *
 * After the parent buffer has been updated, the sub-buffers are implicitly
 * updated as well (we cannot know which parts of the parent was changed),
 * which is marked by setting the sub-buffers to their largest versions.
 *
 * @param updated_buf The updated buffer with its latest_version updated.
 */
static void
update_subbuffer_versioning_data (cl_mem updated_buf)
{
  if (updated_buf->sub_buffers == NULL)
    return;

  cl_mem_list_item_t *sub_buf;
  LL_FOREACH (updated_buf->sub_buffers, sub_buf)
    {
      sub_buf->mem->latest_version = updated_buf->parent->latest_version;
    }
}

/**
 * Creates the necessary implicit migration commands to ensure data is
 * where it's supposed to be according to the semantics of the program
 * defined using commands, buffers, command queues and events.
 *
 * Attempts to use direct copies from a device instead of a device-host-device
 * hip in case there is an accessible peer device with a fresh copy available.
 *
 * @param ev_export_p Optional output parameter for the export event.
 * @param dev Destination device
 * @param user_cmd The event that marks the command that uses the data of the
 * buffer.
 * @param mem The buffer to migrate.
 * @param gmem Identifier of the global memory where the mem should be
 * migrated.
 * @param migration_size Max number of bytes to migrate (caller has to read
 *                       content size from mem->size_buffer if applicable).
 */
int
pocl_create_migration_commands (cl_device_id dev,
                                cl_event *ev_export_p,
                                cl_event user_cmd,
                                cl_mem mem,
                                pocl_mem_identifier *gmem,
                                const char readonly,
                                cl_command_type command_type,
                                cl_mem_migration_flags mig_flags,
                                uint64_t migration_size)
{
  int errcode = CL_SUCCESS;

  cl_event ev_export = NULL, ev_import = NULL, previous_last_event = NULL,
           last_migration_event = NULL;
  _cl_command_node *cmd_export = NULL, *cmd_import = NULL;
  cl_device_id ex_dev = NULL;
  cl_command_queue ex_cq = NULL, dev_cq = NULL;
  int can_directly_mig = 0;
  size_t i;

  POCL_MSG_PRINT_MEMORY ("Analyzing implicit migration of buf %zu %s(latest "
                         "v%zu) to device %zu (has v%zu).\n",
                         mem->id, mem->parent != NULL ? "(sub-buffer) " : "",
                         mem->latest_version, dev->id,
                         mem->device_ptrs[dev->global_mem_id].version);

  /* "export" means copy buffer content from source device to mem_host_ptr;
   *
   * "import" means copy mem_host_ptr content to destination device,
   * or copy directly between devices
   *
   * "need_hostptr" if set, increase the mem_host_ptr_refcount,
   * to keep the mem_host_ptr backing memory around */
  int do_import = 0, do_export = 0, do_need_hostptr = 0;

  /*****************************************************************/

  /* This part only:
   *   sets up the buffer content versions according to requested migration
   * type; sets the buffer->last_updater pointer to the user; decides what
   * needs to be actually done (import, export) but not do it;
   *
   * ... so that any following command sees a correct buffer state.
   * The actual migration commands are enqueued after. */
  POCL_LOCK_OBJ (mem);

  /* Retain the migrated buffer for the duration of the command.
     The symmetric releases as are in pocl_mem_manager_free_command().
  */

  POCL_RETAIN_BUFFER_UNLOCKED (mem);

  /* Save buffer's current last_event as previous last_event,
   * then set the last_event pointer to the actual command's event
   * (user_cmd).
   *
   * We'll need the "previous" event to properly chain commands, but
   * will release it after we've enqueued the required commands. */
  previous_last_event = mem->last_updater;
  mem->last_updater = user_cmd;

  /* Find the device/gmem with the latest copy of the data and that has the
   * fastest migration route.
   * ex_dev = device with the latest copy _other than dev_
   * dev_cq = default command queue for destination dev */
  int highest_d2d_mig_priority = 0;
  for (i = 0; i < mem->context->num_devices; ++i)
    {
      cl_device_id d = mem->context->devices[i];
      cl_command_queue cq = mem->context->default_queues[i];
      if (d == dev)
        dev_cq = cq;
      else if (mem->device_ptrs[d->global_mem_id].version
               == mem->latest_version)
        {
          int cur_d2d_mig_priority = 0;
          if (d->ops->can_migrate_d2d)
            cur_d2d_mig_priority = d->ops->can_migrate_d2d (dev, d);

          /* If we can directly migrate, and we found a better device, use it.
           */
          if (cur_d2d_mig_priority > highest_d2d_mig_priority)
            {
              ex_dev = d;
              ex_cq = cq;
              highest_d2d_mig_priority = cur_d2d_mig_priority;
            }

          /* If we can't migrate D2D, just use plain old through-host
           * migration. */
          if (highest_d2d_mig_priority == 0)
            {
              ex_dev = d;
              ex_cq = cq;
            }
        }
    }

  assert (dev);
  assert (dev_cq);
  /* ex_dev can be NULL, or non-NULL != dev */
  assert (ex_dev != dev);

  /* If mem_host_ptr_version < latest_version, one of devices must have it.
   *
   * could be latest_version == mem_host_ptr_version == some p->version
   * for some p, and so i < ndev; in that case,
   * we leave ex_dev set since D2D is preferred migration way;
   *
   * otherwise must be
   * mem_host_ptr_version == latest_version & > all p->version */

  if ((mem->mem_host_ptr_version < mem->latest_version)
      && (gmem->version != mem->latest_version))
    assert ((ex_dev != NULL)
            && (mem->device_ptrs[ex_dev->global_mem_id].version
                == mem->latest_version));

  /* if ex_dev is NULL, either we have the latest or it's in mem_host_ptr */
  if (ex_dev == NULL)
    assert ((gmem->version == mem->latest_version)
            || (mem->mem_host_ptr_version == mem->latest_version));

  /*****************************************************************/

  /* buffer must be already allocated on this device's globalmem */
  assert (gmem->mem_ptr != NULL);

  /* we're migrating to host mem only: clEnqueueMigMemObjs() with HOST flag */
  if (mig_flags & CL_MIGRATE_MEM_OBJECT_HOST)
    {
      do_import = 0;
      do_export = 0;
      do_need_hostptr = 1;
      if (mem->mem_host_ptr_version < mem->latest_version)
        {
          mem->mem_host_ptr_version = mem->latest_version;
          /* migrate content only if needed */
          if ((mig_flags & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) == 0
              || migration_size == 0)
            {
              /* Could be that destination dev has the latest version,
               * we still need to migrate to host mem */
              if (ex_dev == NULL)
                {
                  ex_dev = dev;
                  ex_cq = dev_cq;
                }
              do_export = 1;
              POCL_RETAIN_OBJECT_UNLOCKED (mem);
            }
        }

      goto FINISH_VER_SETUP;
    }

  /* Otherwise, we're migrating to a device memory. */
  /* Check if we can migrate to the device associated with command_queue
   * without incurring the overhead of migrating their contents */
  if (mig_flags & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
      || migration_size == 0)
    gmem->version = mem->latest_version;

  can_directly_mig = highest_d2d_mig_priority > 0;

  /* Set the flag so the hostptr refcount gets incremented to keep the buffer
   * alive until it has been read for the associated content buffer's
   * migration. */
  if ((mem->content_buffer != NULL) && !can_directly_mig)
    do_need_hostptr = 1;

  /* if we don't need to migrate, skip to end */
  if (gmem->version >= mem->latest_version)
    {
      do_import = 0;
      do_export = 0;
      POCL_MSG_PRINT_MEMORY ("The device has a fresh(er) version of the "
                             "buffer, skipping migration.\n");
      goto FINISH_VER_SETUP;
    }

  /* if mem_host_ptr is outdated AND the devices can't migrate
   * between each other, we need an export command */
  if ((mem->mem_host_ptr_version != mem->latest_version)
      && (can_directly_mig == 0))
    {
      /* we need two migration commands; one on the "source" device's hidden
       * queue, and one on the destination device. */
      do_import = 1;
      do_export = 1;
      do_need_hostptr = 1;

      /* because the two migrate commands will clRelease the buffer */
      POCL_RETAIN_OBJECT_UNLOCKED (mem);
      POCL_RETAIN_OBJECT_UNLOCKED (mem);
      mem->mem_host_ptr_version = mem->latest_version;
      gmem->version = mem->latest_version;
    }
  /* otherwise either:
   * 1) mem_host_ptr is latest, and we need to migrate mem-host-ptr to device,
   * or 2) mem_host_ptr is not latest, but devices can migrate directly between
   * each other, For both cases we only need one migration command on the
   * destination device. */
  else
    {
      do_import = 1;
      do_export = 0;
      do_need_hostptr = 1;

      /* because the corresponding migrate command will clRelease the buffer */
      POCL_RETAIN_OBJECT_UNLOCKED (mem);
      gmem->version = mem->latest_version;
    }

FINISH_VER_SETUP:
  /* If the command is a write-use, increase the version. */
  if (!readonly)
    {
      ++gmem->version;
      mem->latest_version = gmem->version;
      if (mem->sub_buffers != NULL)
        {
          /* A parent buffer update implicitly updates the sub-buffers. Make
             this visible in the gmem buffer versioning info. */
          update_subbuffer_versioning_data (mem);
          cl_mem_list_item_t *sub_buf;
          LL_FOREACH (mem->sub_buffers, sub_buf)
            {
              /* Update the versioning data to the device holding the latest
               * copy. */
              sub_buf->mem->device_ptrs[dev->global_mem_id].version
                = sub_buf->mem->latest_version;
            }
        }
    }

  if (do_need_hostptr)
    {
      /* increase refcount for the two mig commands and for the caller
       * if this is a size buffer needed for content size -aware migration */
      if (do_export)
        ++mem->mem_host_ptr_refcount;
      if (do_import)
        ++mem->mem_host_ptr_refcount;
      if (ev_export_p)
        ++mem->mem_host_ptr_refcount;

      /* allocate mem_host_ptr here if needed... */
      if (mem->mem_host_ptr == NULL)
        {
          size_t align = max (mem->context->min_buffer_alignment, 16);
          /* Always allocate mem_host_ptr for the full size of the buffer to
           * guard against applications forgetting to check content size. */
          mem->mem_host_ptr = pocl_aligned_malloc (align, mem->size);
          assert ((mem->mem_host_ptr != NULL)
                  && "Cannot allocate backing memory for mem_host_ptr!\n");
        }
    }

  /*****************************************************************/

  /* Enqueue a command for export.
   * Put the previous last event into its waitlist. */
  if (do_export)
    {
      assert (ex_cq);
      assert (ex_dev);
      errcode = pocl_create_command_struct (
        &cmd_export, ex_cq, CL_COMMAND_MIGRATE_MEM_OBJECTS,
        &ev_export, // event_p
        (previous_last_event ? 1 : 0),
        (previous_last_event ? &previous_last_event : NULL), // waitlist
        NULL);
      assert (errcode == CL_SUCCESS);
      if (do_need_hostptr)
        ev_export->release_mem_host_ptr_after = 1;

      cmd_export->command.migrate.type = ENQUEUE_MIGRATE_TYPE_D2H;
      cmd_export->command.migrate.implicit = 1;
      cmd_export->command.migrate.migration_size = migration_size;
      cmd_export->command.migrate.num_buffers = 1;
      cmd_export->migr_infos
        = pocl_append_unique_migration_info (NULL, mem, 0);

      last_migration_event = ev_export;

      if (ev_export_p)
        {
          POname (clRetainEvent) (ev_export);
          *ev_export_p = ev_export;
        }
    }

  /* enqueue a command for import.
   * Put either the previous last event, or export ev, into its waitlist. */
  if (do_import)
    {
      /* the import command must depend on (wait for) either the export
       * command, or the buffer's previous last event. Can be NULL if there's
       * no last event or export command */
      cl_event import_wait_ev = (ev_export ? ev_export : previous_last_event);

      errcode = pocl_create_command_struct (
        &cmd_import, dev_cq, CL_COMMAND_MIGRATE_MEM_OBJECTS,
        &ev_import, // event_p
        (import_wait_ev ? 1 : 0),
        (import_wait_ev ? &import_wait_ev : NULL), // waitlist
        NULL);
      assert (errcode == CL_SUCCESS);
      if (do_need_hostptr)
        ev_import->release_mem_host_ptr_after = 1;

      if (can_directly_mig)
        {
          cmd_import->command.migrate.type = ENQUEUE_MIGRATE_TYPE_D2D;
          cmd_import->command.migrate.implicit = 1;
          cmd_import->command.migrate.src_device = ex_dev;

          if (mem->size_buffer != NULL)
            cmd_import->command.migrate.src_content_size_mem_id
              = &mem->size_buffer->device_ptrs[ex_dev->global_mem_id];
        }
      else
        {
          cmd_import->command.migrate.type = ENQUEUE_MIGRATE_TYPE_H2D;
          cmd_import->command.migrate.implicit = 1;
          cmd_import->command.migrate.migration_size = migration_size;
        }
      cmd_import->command.migrate.num_buffers = 1;
      cmd_import->migr_infos
        = pocl_append_unique_migration_info (NULL, mem, 0);

      /* because explicit event */
      if (ev_export)
        POname (clReleaseEvent) (ev_export);

      last_migration_event = ev_import;
    }

  /* we don't need it anymore. */
  if (previous_last_event)
    POname (clReleaseEvent (previous_last_event));

  /* the final event must depend on the export/import commands */
  if (last_migration_event)
    {
      pocl_create_event_sync (user_cmd, last_migration_event);
      /* if the event itself only reads from the buffer,
       * set the last buffer updating event to the last_mig_event,
       * instead of the actual command event;
       * this avoids unnecessary waits e.g on kernels
       * which only read from buffers */
      if (readonly)
        {
          mem->last_updater = last_migration_event;
          POname (clReleaseEvent) (user_cmd);
        }
      else /* because explicit event */
        POname (clReleaseEvent) (last_migration_event);
    }
  POCL_UNLOCK_OBJ (mem);

  if (do_export)
    {
      POCL_MSG_PRINT_MEMORY (
        "Queuing a %zu-byte device-to-host migration for buf %zu%s\n",
        migration_size, mem->id, mem->parent != NULL ? " (sub-buffer)" : "");

      pocl_command_enqueue (ex_cq, cmd_export);
    }

  if (do_import)
    {
      POCL_MSG_PRINT_MEMORY (
        "Queuing a %zu-byte host-to-device migration for buf %zu%s\n",
        migration_size, mem->id, mem->parent != NULL ? " (sub-buffer)" : "");

      pocl_command_enqueue (dev_cq, cmd_import);
    }

  return CL_SUCCESS;
}

/**
 * Creates a deep copy of the migration info list.
 *
 * Also retains the memobjects so they can be released per list reference.
 */
pocl_buffer_migration_info *
pocl_deep_copy_migration_info_list (pocl_buffer_migration_info *list)
{
  pocl_buffer_migration_info *new_list = NULL;
  pocl_buffer_migration_info *mi;
  LL_FOREACH (list, mi)
    {
      POname (clRetainMemObject (mi->buffer));
      new_list = pocl_append_unique_migration_info (new_list, mi->buffer,
                                                    mi->read_only);
    }
  return new_list;
}

/**
 * Appends a migration info to the given list, if it doesn't already have
 * one with the given buffer.
 *
 * Creates the list if it's empty. If the buffer is NULL, doesn't modify
 * the list. Upgrades an existing read-only buffer usage to a read-write, if
 * the appended buffer writes to a buffer that is already in the lst.
 *
 * @return Returns a pointer to the beginning of the list.
 */
pocl_buffer_migration_info *
pocl_append_unique_migration_info (pocl_buffer_migration_info *list,
                                   cl_mem buffer,
                                   char read_only)
{
  if (buffer == NULL)
    return list;
  pocl_buffer_migration_info *found_info = NULL;
  if (list != NULL)
    {
      LL_FOREACH (list, found_info)
        {
          if (found_info->buffer == buffer)
            {
              if (found_info->read_only && !read_only)
                found_info->read_only = 0;
              return list;
            }
        }
    }

  pocl_buffer_migration_info *migr_info
    = (pocl_buffer_migration_info *)calloc (
      1, sizeof (pocl_buffer_migration_info));
  migr_info->buffer = buffer;
  migr_info->read_only = read_only;
  DL_APPEND (list, migr_info);
  return list;
}
