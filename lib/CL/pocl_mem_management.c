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

static cl_mem
add_unique_implicit_sub_buffer (cl_mem parent,
                                size_t origin,
                                size_t size,
                                cl_int *retv)
{
  size_t align = parent->context->mem_base_addr_align;
  size_t aligned_origin = ((origin + align - 1) / align) * align;

  size_t aligned_size = size - (aligned_origin - origin);

  *retv = CL_SUCCESS;
  /* Due to the round-up of the starting address, we might end up with
     an empty sub-buffer. In that case we rely on the pre-updates to
     fill the spots. */
  if (aligned_origin - origin >= size || aligned_origin >= parent->size)
    return NULL;

  if (aligned_origin + aligned_size > parent->size)
    aligned_size = parent->size - aligned_origin;

  POCL_LOCK_OBJ (parent);
  cl_mem_list_item_t *sb;
  LL_FOREACH (parent->implicit_sub_buffers, sb)
    {
      if (sb->mem->origin == aligned_origin && sb->mem->size == aligned_size)
        {
          /* Found one already, nothing to do. */
          POCL_UNLOCK_OBJ (parent);
          return sb->mem;
        }
    }
  POCL_UNLOCK_OBJ (parent);

  cl_buffer_region Region = { .origin = aligned_origin, .size = aligned_size };
  cl_int err;

  cl_mem sub_buf = POname (clCreateSubBuffer) (
    parent, 0, CL_BUFFER_CREATE_TYPE_REGION, &Region, &err);

  if (err != CL_SUCCESS)
    {
      POCL_MSG_PRINT_MEMORY ("Unable to create an implicit sub-buffer.\n");
      *retv = err;
      return NULL;
    }
  sub_buf->implicit_sub_buffer = 1;

  /* The implicit sub-buffers should not keep the parent buffer
     or the context alive, but just get cleaned up silently when
     the parent buffer is (in clReleaseBuffer()). Release the refs
     created in clCreateSubBuffer() here. */
  int newrefc;
  POCL_RELEASE_OBJECT (sub_buf->parent, newrefc);
  assert (newrefc > 0);
  POCL_RELEASE_OBJECT (sub_buf->context, newrefc);
  assert (newrefc > 0);

  cl_mem_list_item_t *sb_item
    = (cl_mem_list_item_t *)calloc (1, sizeof (cl_mem_list_item_t));

  sb_item->mem = sub_buf;

  POCL_LOCK_OBJ (parent);
  LL_APPEND (parent->implicit_sub_buffers, sb_item);
  POCL_UNLOCK_OBJ (parent);

  POCL_MSG_PRINT_MEMORY ("Created an implicit sub-buffer %zu to cover %zu "
                         "bytes from offset %zu\n",
                         sub_buf->id, aligned_size, aligned_origin);
  return sub_buf;
}

/**
 * Generates "implicit" sub-buffers that cover the parts of the parent
 * buffer that are not covered by user-created sub-buffers.
 *
 * The implicit sub-buffers are used to implicitly synchronize the parent
 * buffer's uncovered parts when accessing the parent buffer after modifying
 * the sub-buffer. The implicit sub-buffers are just like user created
 * sub-buffers (created with clCreateSubBuffers()) and are stored inside
 * the parent cl_mem with lifetimes tied to it. There is currently no
 * garbage collection on the implicit subbuffers, so if the parent buffer
 * will be split in numerous ways by the client, it will just add new
 * implicit sub-buffers to cover the new empty spots which are previously
 * not covered.
 *
 * The implicit sub-buffers generated by this function are aligned to the
 * maximum mem base addr align. This means that in case there are user
 * sub-buffers which before unaligned addresses, there are unmigrated spots
 * that need to be covered with additional sub-buffers that must be migrated
 * *before* the user sub-buffers, which will update the sub-buffer covered
 * parts.
 *
 */
static cl_int
generate_implicit_aligned_sub_buffers (cl_mem parent,
                                       pocl_buffer_migration_info *migrations,
                                       int read_only)
{
  struct buf_usage
  {
    size_t offset;
    size_t size;
    struct buf_usage *prev, *next;
  };

  assert (migrations != NULL);

  cl_int retv = CL_SUCCESS;

  if (parent->sub_buffers == NULL)
    return CL_SUCCESS;

  /* A list of sorted buffer address range usages. If there are overlapping
     sub-buffers, they get merged to a single buf_usage. */
  struct buf_usage *usage = NULL;
  cl_mem_list_item_t *sub_buf;
  LL_FOREACH (parent->sub_buffers, sub_buf)
    {
      /* We might have generated implicit sub-buffers before, but
         the user might have added new sub-buffers after that,
         so we must process the whole list. */
      if (sub_buf->mem->implicit_sub_buffer)
        continue;

      /* Find if there's an overlapping buffer and if not, add a new one
         before/after it, depending on the position. If found, merge to an
         overlapping existing buffer right away. */

      /* Yeah, it's unoptimal. Hopefully there won't be zillions of
       * sub-buffers. */
      struct buf_usage *buf_usage = NULL, *spot_before = NULL;
      int merged = 0;
      LL_FOREACH (usage, buf_usage)
        {
          if (buf_usage->offset <= sub_buf->mem->origin
              && buf_usage->offset + buf_usage->size >= sub_buf->mem->origin)
            {
              /* Expand the buffer usage with the overlapping part (if any).
                 Otherwise there's full overlap, thus it gets absorbed to this
                 usage. */
              if (sub_buf->mem->origin + sub_buf->mem->size
                  > buf_usage->offset + buf_usage->size)
                buf_usage->size += sub_buf->mem->origin + sub_buf->mem->size
                                   - (buf_usage->offset + buf_usage->size);
              merged = 1;
              break;
            }
          else if (buf_usage->offset
                   > sub_buf->mem->origin + sub_buf->mem->size)
            {
              /* This usage is already past the one we are inserting, prepend.
               */
              spot_before = buf_usage;
              break;
            }
          /* Otherwise we will end up with NULL, which means we are adding a
             new largest one. */
        }
      if (merged)
        break;

      struct buf_usage *new_usage = calloc (1, sizeof (struct buf_usage));
      new_usage->size = sub_buf->mem->size;
      new_usage->offset = sub_buf->mem->origin;
      if (spot_before != NULL)
        {
          new_usage->next = spot_before;
          new_usage->prev = spot_before->prev;
          if (spot_before->prev == NULL)
            usage = new_usage;
          spot_before->prev = new_usage;
        }
      else
        {
          DL_APPEND (usage, new_usage);
        }
    }

  size_t align = parent->context->mem_base_addr_align;
  /* Create/find the implicit sub-buffers for the empty spots. */
  struct buf_usage *bu = NULL, *tmp = NULL;
  size_t pos = 0;
  LL_FOREACH (usage, bu)
    {
      size_t chunk_offset = pos;
      size_t chunk_size = bu->offset - pos;
      pos = bu->offset + bu->size;
      if (chunk_size == 0)
        /* End-to-start sub-buffer. No gap. */
        continue;

      cl_mem sb = add_unique_implicit_sub_buffer (parent, chunk_offset,
                                                  chunk_size, &retv);
      pocl_append_unique_migration_info (migrations, sb, read_only);

      if (retv != CL_SUCCESS)
        goto out;
    }
  /* Check if there's uncovered space in the end of the buffer. */
  size_t chunk_size = parent->size - pos;
  if (chunk_size > 0)
    {
      cl_mem sb
        = add_unique_implicit_sub_buffer (parent, pos, chunk_size, &retv);
      pocl_append_unique_migration_info (migrations, sb, read_only);
    }

out:
  LL_FOREACH_SAFE (usage, bu, tmp)
    free (bu);
  return retv;
}

static void
pocl_dump_migration_infos (pocl_buffer_migration_info *mis)
{
  pocl_buffer_migration_info *mi = NULL;
  LL_FOREACH (mis, mi)
    {
      if (mi->buffer->parent == NULL)
        fprintf (stderr, "buf %zu latest v%zu host v%zu%s\n", mi->buffer->id,
                 mi->buffer->latest_version, mi->buffer->mem_host_ptr_version,
                 mi->read_only ? " (ro)" : "");
      else
        fprintf (stderr,
                 "sbuf#%zu/#%zu [%zu ... %zu] latest v%zu host v%zu%s parent "
                 "latest v%zu\n",
                 mi->buffer->id, mi->buffer->parent->id, mi->buffer->origin,
                 mi->buffer->origin + mi->buffer->size - 1,
                 mi->buffer->latest_version, mi->buffer->mem_host_ptr_version,
                 mi->read_only ? " (ro)" : "",
                 mi->buffer->parent->latest_version);
    }
}

/** Creates an implicit sub-buffer to patch up a part left by another
   sub-buffers which ends before an unaligned address.

   For example, assuming align of 8 bytes (X = covered address):

   USER SB:            XXXXXXXX XXX-----

   Here the SB ends at address 10. The aligned implicit SB starts at 16
   as its start address is rounded up to the next aligned address:

   ALIGNED IMPLICT SB: -------- -------- XXX...

   So, we need to pre-update addresses [8...15] from the freshest parent
   buffer copy with an aligned transfer that is done *before* the USB
   migration, letting it first write the parts that are not covered by
   the user SB, and then letting the USB to update new data at [8..10].

   PRE-IMPLICIT SB:    -------- XXXXXXXX ---...

   @param patches The list of migrations to (possibly) append to.
   @param sub_buf The (possibly) unaligned sub buffer to patch up.
   @param align The alignment to aim for.
*/
static pocl_buffer_migration_info *
append_unaligned_patch_subbuffer_migration (
  pocl_buffer_migration_info *patches,
  cl_mem sub_buf,
  size_t align,
  int read_only,
  cl_int *retv)
{
  size_t next_start_addr = sub_buf->origin + sub_buf->size;
  assert (sub_buf->origin % align == 0);
  *retv = CL_SUCCESS;
  if (next_start_addr % align == 0)
    return patches;

  /* Create an aligned sub-buffer covering just the uncovered part as
     illustrated in the function comment. */
  next_start_addr = (next_start_addr / align) * align;
  cl_mem sb = add_unique_implicit_sub_buffer (sub_buf->parent, next_start_addr,
                                              align, retv);
  if (*retv != CL_SUCCESS)
    return NULL;
  POCL_MSG_PRINT_MEMORY ("Sub-buffer %zu is a pre-patch to update %zu "
                         "bytes from offset %zu due to sb %zu\n",
                         sb->id, sb->size, sb->origin, sub_buf->id);
  return pocl_append_unique_migration_info (patches, sb, read_only);
}

/* Splits migrations of parent buffers to sub-buffer migrations, if the buffer
 * has sub-buffers, otherwise does nothing.
 *
 * Use implicitly generated sub-buffers to cover the synchronization of the
 * uncovered parts of the parent buffer, if any.
 *
 * @param buffer_usage The original buffer usage. The sub-buffer migrations are
 * added to it in the order of required migrations.
 **/
pocl_buffer_migration_info *
pocl_convert_to_subbuffer_migrations (pocl_buffer_migration_info *buffer_usage,
                                      cl_int *err)
{
  pocl_buffer_migration_info *extra_migrations = NULL;
  pocl_buffer_migration_info *patch_migrations = NULL;

#if 0
  fprintf (stderr, "Original migrations:\n");
  pocl_dump_migration_infos (buffer_usage);
#endif

  pocl_buffer_migration_info *mi, *tmp = NULL;
  LL_FOREACH_SAFE (buffer_usage, mi, tmp)
    {
      size_t align = mi->buffer->context->mem_base_addr_align;
      cl_int retv;
      if (mi->buffer->parent != NULL)
        {
          /* This is a sub-buffer. Just ensure we migrate a possible unaligned
             end part of it also by prepending a patch-up SB. Check
             append_unaligned_patch_subbuffer_migration() for more docs. */
          patch_migrations = append_unaligned_patch_subbuffer_migration (
            patch_migrations, mi->buffer, align, mi->read_only, &retv);
          if (retv != CL_SUCCESS)
            return NULL;
          continue;
        }
      else if (mi->buffer->sub_buffers == NULL)
        continue;

      /* Generate implicit sub-buffers for migrating the parent buffer from
         pieces. */

      cl_mem_list_item_t *sub_buf;
      LL_FOREACH (mi->buffer->sub_buffers, sub_buf)
        {
          /* First add migrations of the user SBs. */
          if (sub_buf->mem->implicit_sub_buffer)
            continue;

          /* TODO: We could here utilize the page fault notification info
             and only migrate the sub-buffers that have been actually
             changed after the previous migration was done. */
          extra_migrations = pocl_append_unique_migration_info (
            extra_migrations, sub_buf->mem, mi->read_only);
        }

      /* Then generate implicit sub-buffers to the end of the sub-buffer list
         which cover the parts not touched by the user sub-buffers and
         will be migrated after the user sub-buffers. */
      if (generate_implicit_aligned_sub_buffers (mi->buffer, extra_migrations,
                                                 mi->read_only)
          != CL_SUCCESS)
        {
          *err = CL_OUT_OF_RESOURCES;
          return NULL;
        }

      /* Finally patch up the parts left by user sub-buffers which end before
         an unaligned address by injecting aligned sub-buffer migrations before
         the actual user SB migrations. */
      LL_FOREACH (mi->buffer->sub_buffers, sub_buf)
        {
          if (sub_buf->mem->implicit_sub_buffer)
            continue;

          patch_migrations = append_unaligned_patch_subbuffer_migration (
            patch_migrations, sub_buf->mem, align, mi->read_only, &retv);
          if (retv != CL_SUCCESS)
            return NULL;
        }

      LL_DELETE (buffer_usage, mi);
      free (mi);
    }
  LL_CONCAT (patch_migrations, buffer_usage);
  LL_CONCAT (patch_migrations, extra_migrations);

#if 0
  fprintf (stderr, "Updated sub-buffer-based migrations:\n");
  pocl_dump_migration_infos (patch_migrations);
#endif

  /* We might replace all of the buffer usages, thus need to return the
     new list head. */
  return patch_migrations;
}

/**
 * Set the sub-buffers of the parent buffer after a parent buffer update.
 *
 * After the parent buffer has been updated, the sub-buffers are implicitly
 * updated as well (we cannot know which parts of the parent was changed),
 * which is marked by setting the sub-buffers to their largest versions.
 *
 * @todo this might not be needed anymore because if there are sub-buffers
 * for a buffer, all of the contents will be managed using implicit and
 * user sub-buffers.
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
 * hop in case there is an accessible peer device with a fresh copy available.
 *
 * \param ev_export_p Optional output parameter for the export event.
 * \param dev Destination device
 * \param user_cmd The event that marks the command that uses the data of the
 * buffer.
 * \param mem The buffer to migrate.
 * \param gmem Identifier of the global memory where the mem should be
 * migrated.
 * \param migration_size Max number of bytes to migrate (caller has to read
 *                       content size from mem->size_buffer if applicable).
 * \param last_migr_event Input/output for dep-chaining the created migration
 * command an event dependency is created to it if non-NULL. The new migration
 * command will be overwritten to it (after reference release).
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
                                uint64_t migration_size,
                                cl_event *prev_migr_event)
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
                         "v%zu) to device %zu (has v%zu) host has v%zu.\n",
                         mem->id, mem->parent != NULL ? "(sub-buffer) " : "",
                         mem->latest_version, dev->id,
                         mem->device_ptrs[dev->global_mem_id].version,
                         mem->mem_host_ptr_version);

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
     The symmetric releases are in pocl_mem_manager_free_command().
  */

  POCL_RETAIN_BUFFER_UNLOCKED (mem);

  /* Save the buffer's current last_event as previous last_event,
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

      /* Because of the explicit event. */
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
      if (prev_migr_event != NULL)
        {
          /* By default serialize all the migrations to avoid races with the
             patch-up migrations that should be written before the other
             migrations. TO OPTIMIZE: Drop unnecessary event deps
             if there is no overlap in order to enable per cmd parallel
             migrations. */
          /* Create an event dep chain through the migration commands. */
          POname (clRetainEvent) (last_migration_event);
          pocl_create_event_sync (*prev_migr_event, last_migration_event);
          if (*prev_migr_event != NULL)
            POname (clReleaseEvent) (*prev_migr_event);
          *prev_migr_event = last_migration_event;
        }
      pocl_create_event_sync (last_migration_event, user_cmd);
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
 * @param retain If set to non-zero, also retains the memobjects so they can
 * be released per list reference.
 */
pocl_buffer_migration_info *
pocl_deep_copy_migration_info_list (pocl_buffer_migration_info *list,
                                    int retain)
{
  pocl_buffer_migration_info *new_list = NULL;
  pocl_buffer_migration_info *mi;
  LL_FOREACH (list, mi)
    {
      if (retain)
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

pocl_raw_ptr *
pocl_find_raw_ptr_with_vm_ptr (cl_context context, const void *host_ptr)
{
  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *item = NULL;
  DL_FOREACH (context->raw_ptrs, item)
    {
      if (item->vm_ptr == NULL)
        continue;
      if (item->vm_ptr <= host_ptr
          && (char *)item->vm_ptr + item->size > (const char *)host_ptr)
        {
          break;
        }
    }
  POCL_UNLOCK_OBJ (context);
  return item;
}

pocl_raw_ptr *
pocl_find_raw_ptr_with_dev_ptr (cl_context context, const void *dev_ptr)
{
  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *item = NULL;
  DL_FOREACH (context->raw_ptrs, item)
    {
      if (item->dev_ptr == NULL)
        continue;
      if (item->dev_ptr <= dev_ptr
          && (char *)item->dev_ptr + item->size > (const char *)dev_ptr)
        break;
    }
  POCL_UNLOCK_OBJ (context);
  return item;
}
