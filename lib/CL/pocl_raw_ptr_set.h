/* Provides container for mapping VM and BDA addresses to pocl_raw_ptr
   objects.

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
#ifndef POCL_RAW_PTR_SET_H
#define POCL_RAW_PTR_SET_H

#include "pocl_raw_ptr.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct pocl_raw_ptr_set pocl_raw_ptr_set;

/** Creates new empty pocl_raw_ptr_set.
 */
POCL_EXPORT
pocl_raw_ptr_set *pocl_raw_ptr_set_create (void);

/** Destroys the pocl_raw_ptr_set object and contained pocl_raw_ptr
 * instances.
 */
POCL_EXPORT
void pocl_raw_ptr_set_destroy (pocl_raw_ptr_set *set);

/** Insert new pocl_raw_ptr object to the set
 *
 * Returns non-zero value is the insertion took place. The set takes
 * ownership of the inserted object. */
POCL_EXPORT
int pocl_raw_ptr_set_insert (pocl_raw_ptr_set *set, pocl_raw_ptr *ptr);

/** Return pocl_raw_ptr object corresponding to the 'vm_ptr'
 *
 * Returns NULL if object was not found.
 */
POCL_EXPORT
pocl_raw_ptr *pocl_raw_ptr_set_lookup_with_vm_ptr (pocl_raw_ptr_set *set,
                                                   const void *vm_ptr);

/** Return pocl_raw_ptr object corresponding to the 'dev_ptr' for the
 * device 'dev'
 *
 * Returns NULL if object was not found.
 */
POCL_EXPORT
pocl_raw_ptr *pocl_raw_ptr_set_lookup_with_dev_ptr (pocl_raw_ptr_set *set,
                                                    cl_device_id dev,
                                                    const void *dev_ptr);

/** Return the head of the set iterable with utlist's DL_FOREACH macro.
 *
 * Behavior is undefined if the items are modified.
 */
POCL_EXPORT
pocl_raw_ptr *pocl_raw_ptr_set_begin (pocl_raw_ptr_set *set);

/** Removes the given 'ptr' from the 'set' but does not delete the
 *  'ptr'.
 */
POCL_EXPORT void pocl_raw_ptr_set_remove (pocl_raw_ptr_set *set,
                                          pocl_raw_ptr *ptr);

/** Removes the given 'ptr' from the 'set' and deletes the 'ptr'.
 */
POCL_EXPORT
void pocl_raw_ptr_set_erase (pocl_raw_ptr_set *set, pocl_raw_ptr *ptr);

/** Deletes all items in the 'set'.
 */
POCL_EXPORT
void pocl_raw_ptr_set_erase_all (pocl_raw_ptr_set *set);

/** Deletes all items in the set with `pocl_raw_ptr::shadow_cl_mem
 *  == shadow_cl_mem`.
 */
POCL_EXPORT
void pocl_raw_ptr_set_erase_all_by_shadow_mem (pocl_raw_ptr_set *Set,
                                               cl_mem shadow_cl_mem);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCL_RAW_PTR_SET_H */
