/* Implementation of pocl_raw_ptr_set.

   Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy

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

#include "pocl_raw_ptr_set.h"

typedef struct
{
  const void *base;
  size_t len;
} addr_range_t;

typedef struct
{
  addr_range_t addr_range;
  cl_device_id dev;
} bda_range_t;

/* Custom compare function for sortedmap to support lookups with
 * possibly offseted pointers. Overlapping address ranges map to same
 * item in the sortedmap. Beware that the behavior is undefined if an item
 * is inserted or looked up with a key (address range) overlapping with two
 * or more other keys in the sortedmap. */
static int
addr_range_cmp (const addr_range_t *lhs, const addr_range_t *rhs)
{
  const char *lhs_base = (const char *)lhs->base;
  const char *rhs_base = (const char *)rhs->base;

  if (lhs_base + lhs->len <= rhs_base)
    return -1;
  if (rhs_base + rhs->len <= lhs_base)
    return 1;
  return 0;
}

static int
bda_range_cmp (const bda_range_t *lhs, const bda_range_t *rhs)
{
  if (lhs->dev < rhs->dev)
    return -1;
  if (lhs->dev > rhs->dev)
    return 1;
  return addr_range_cmp (&lhs->addr_range, &rhs->addr_range);
}

#define T vm_ptr_map_t, addr_range_t, pocl_raw_ptr *
#define i_cmp addr_range_cmp
#include <stc/sortedmap.h>

#define T bda_ptr_map_t, bda_range_t, pocl_raw_ptr *
#define i_cmp bda_range_cmp
#include <stc/sortedmap.h>

#include "pocl_cl.h"
#include "utlist.h"

struct pocl_raw_ptr_set
{
  pocl_raw_ptr *head;
  vm_ptr_map_t vm_ptr_map;
  bda_ptr_map_t bda_ptr_map;
};

void
pocl_raw_ptr_check_invariant (const pocl_raw_ptr *ptr)
{
  assert (ptr && "Invalid pocl_raw_ptr argument!");
  assert ((ptr->vm_ptr != ptr->dev_ptr || !ptr->vm_ptr)
          && "vm_ptr and dev_ptr can't be both set");
  assert (ptr->size && "Zero size allocation!");

  if (ptr->dev_ptr)
    assert (ptr->device); /* Needed to disambiguate aliasing BDA addresses. */

  /* 'ptr' can't be already be inserted in another container -
     pocl_raw_ptr_set will take the ownership of the 'ptr'.  */
  assert (!ptr->prev);
  assert (!ptr->next);
}

POCL_EXPORT pocl_raw_ptr_set *
pocl_raw_ptr_set_create ()
{
  return calloc (1, sizeof (pocl_raw_ptr_set));
}

POCL_EXPORT void
pocl_raw_ptr_set_destroy (pocl_raw_ptr_set *set)
{
  if (!set)
    return;
  pocl_raw_ptr_set_erase_all (set);
  vm_ptr_map_t_drop (&set->vm_ptr_map);
  bda_ptr_map_t_drop (&set->bda_ptr_map);
  free (set);
}

POCL_EXPORT int
pocl_raw_ptr_set_insert (pocl_raw_ptr_set *set, pocl_raw_ptr *raw_ptr)
{
  assert (set && "Invalid pocl_raw_ptr_set argument!");
  pocl_raw_ptr_check_invariant (raw_ptr);

  if (raw_ptr->vm_ptr)
    {
      addr_range_t key = { raw_ptr->vm_ptr, raw_ptr->size };
      vm_ptr_map_t_result result
        = vm_ptr_map_t_insert (&set->vm_ptr_map, key, raw_ptr);
      assert (result.inserted && "Overlapping VM pointer!");
      (void)result;
    }

  if (raw_ptr->dev_ptr)
    {
      bda_range_t key
        = { { raw_ptr->dev_ptr, raw_ptr->size }, raw_ptr->device };
      bda_ptr_map_t_result result
        = bda_ptr_map_t_insert (&set->bda_ptr_map, key, raw_ptr);
      assert (result.inserted && "Overlapping BDA pointer!");
      (void)result;
    }

  DL_APPEND (set->head, raw_ptr);

  return 1;
}

POCL_EXPORT pocl_raw_ptr *
pocl_raw_ptr_set_lookup_with_vm_ptr (pocl_raw_ptr_set *set, const void *vm_ptr)
{
  assert (set && "Invalid pocl_raw_ptr_set argument!");
  addr_range_t key = { vm_ptr, 1 };
  vm_ptr_map_t_iter it = vm_ptr_map_t_find (&set->vm_ptr_map, key);
  return it.ref ? it.ref->second : NULL;
}

POCL_EXPORT pocl_raw_ptr *
pocl_raw_ptr_set_lookup_with_dev_ptr (pocl_raw_ptr_set *set,
                                      cl_device_id dev,
                                      const void *dev_ptr)
{
  assert (set && "Invalid pocl_raw_ptr_set argument!");
  bda_range_t key = { { dev_ptr, 1 }, dev };
  bda_ptr_map_t_iter it = bda_ptr_map_t_find (&set->bda_ptr_map, key);
  return it.ref ? it.ref->second : NULL;
}

POCL_EXPORT pocl_raw_ptr *
pocl_raw_ptr_set_begin (pocl_raw_ptr_set *set)
{
  assert (set && "invalid pocl_raw_ptr_set argument!");
  return set->head;
}

POCL_EXPORT void
pocl_raw_ptr_set_remove (pocl_raw_ptr_set *set, pocl_raw_ptr *ptr)
{
  assert (set && "Invalid pocl_raw_ptr_set argument!");

  if (!ptr)
    return;

#ifndef NDEBUG
  int membership = 0;
  pocl_raw_ptr *temp_ptr;
  DL_FOREACH (set->head, temp_ptr)
    {
      membership = temp_ptr == ptr;
      if (membership)
        break;
    }
  assert (membership && "'ptr' is not a member of pocl_raw_ptr_set.");
#endif

  if (ptr->vm_ptr)
    {
      addr_range_t key = { ptr->vm_ptr, 1 };
      vm_ptr_map_t_erase (&set->vm_ptr_map, key);
    }

  if (ptr->dev_ptr)
    {
      bda_range_t key = { { ptr->dev_ptr, 1 }, ptr->device };
      bda_ptr_map_t_erase (&set->bda_ptr_map, key);
    }

  DL_DELETE (set->head, ptr);
}

POCL_EXPORT void
pocl_raw_ptr_set_erase (pocl_raw_ptr_set *set, pocl_raw_ptr *ptr)
{
  pocl_raw_ptr_set_remove (set, ptr);
  free (ptr);
}

POCL_EXPORT void
pocl_raw_ptr_set_erase_all (pocl_raw_ptr_set *set)
{
  assert (set && "Invalid pocl_raw_ptr_set argument!");

  pocl_raw_ptr *ptr = NULL;
  pocl_raw_ptr *ptr_next = NULL;
  DL_FOREACH_SAFE (set->head, ptr, ptr_next)
    {
      free (ptr);
    }

  set->head = NULL;
  vm_ptr_map_t_clear (&set->vm_ptr_map);
  bda_ptr_map_t_clear (&set->bda_ptr_map);
}

POCL_EXPORT void
pocl_raw_ptr_set_erase_all_by_shadow_mem (pocl_raw_ptr_set *set,
                                          cl_mem shadow_mem)
{
  assert (set && "Invalid pocl_raw_ptr_set argument!");
  assert (shadow_mem && "Invalid cl_mem argument!");

  /* Linear probe. This can be improved later if needed. */
  pocl_raw_ptr *ptr = NULL;
  pocl_raw_ptr *tmp = NULL;
  DL_FOREACH_SAFE (set->head, ptr, tmp)
    {
      if (ptr->shadow_cl_mem == shadow_mem)
        pocl_raw_ptr_set_erase (set, ptr);
    }
}
