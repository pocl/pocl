/* OpenCL runtime/device driver library: custom buffer allocator

   Copyright (c) 2011 Tampere University of Technology

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
/**
 * This file implements a customized memory allocator for OpenCL buffers.
 *
 * @see bufalloc.c
 * @file bufalloc.h
 */

#ifndef BUFALLOC_H
#define BUFALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#ifndef __TCE_STANDALONE__

#include "pocl_cl.h"

typedef pocl_lock_t ba_lock_t;

#define BA_LOCK(LOCK) POCL_LOCK(LOCK)
#define BA_UNLOCK(LOCK) POCL_UNLOCK(LOCK)
#define BA_INIT_LOCK(LOCK) POCL_INIT_LOCK(LOCK)

/* The qualifier to add in case using non-default
   address space. */
#ifndef AS_QUALIFIER
#define AS_QUALIFIER
#endif

#else
/* TCE standalone mode. */

#include "tce-sync.h"
#include "dthread.h"

#ifdef NO_LOCKING
typedef int ba_lock_t;
#define BA_LOCK(LOCK)
#define BA_UNLOCK(LOCK)
#define BA_INIT_LOCK(LOCK)
#else
typedef tce_sm_lock ba_lock_t;
#define BA_LOCK(LOCK) tce_sync_spin_lock (&LOCK)
#define BA_UNLOCK(LOCK) tce_sync_unlock (&LOCK)
#define BA_INIT_LOCK(LOCK) LOCK = 0
#endif

#ifndef AS_QUALIFIER
#define AS_QUALIFIER
#endif

#endif

/* The number of chunks in a region should be scaled to an approximate
   maximum number of kernel buffer arguments. Running out of chunk
   data structures might leave region space unused due to that only. */
#ifndef MAX_CHUNKS_IN_REGION
#define MAX_CHUNKS_IN_REGION 1024
#endif

/* address-space agnostic memory address */
typedef size_t memory_address_t;

/* the different strategies for how to allocate buffers from a memory region */
enum allocation_strategy
  {
    BALLOCS_WASTEFUL, /* try to fit to the end of the region first
                         (consumes the whole region quicker) */
    BALLOCS_TIGHT     /* try to reuse old freed chunks first
                         (for the case when the region grows dynamically e.g. towards stack)
                      */
  };

#ifdef __TCE_STANDALONE__
typedef AS_QUALIFIER volatile struct chunk_info chunk_info_t;
typedef AS_QUALIFIER volatile struct memory_region memory_region_t;
#else
typedef AS_QUALIFIER struct chunk_info chunk_info_t;
typedef AS_QUALIFIER struct memory_region memory_region_t;
#endif

/* Info of a single "chunk" inside a memory region. Chunk is a piece
   of memory that has been allocated to a buffer (but might have been
   unallocated). Initially there's only one special chunk representing
   the whole region as one unallocated chunk. */
struct chunk_info
{
  memory_address_t start_address;
  int is_allocated;
  size_t size; /* size in bytes */
  chunk_info_t* next;
  chunk_info_t* prev;
  chunk_info_t* children;
  chunk_info_t* parent;
  memory_region_t* parent_region;
};

/* Represents a single continuous region of memory from which smaller
   "chunks" are allocated. Note: this doesn't include the memory space
   itself. */
struct memory_region
{
  chunk_info_t all_chunks[MAX_CHUNKS_IN_REGION];
  chunk_info_t *chunks;
  chunk_info_t *free_chunks; /* A pointer to a head of a linked list of
                                chunk_info records that can be used for
                                new allocations. This enables allocating
                                the chunk infos statically at compile time,
                                or dynamically. In the dynamic case, the
                                client of the bufalloc should first ensure
                                there is at least one free chunk info before
                                trying the allocation. If not, create one. */
  chunk_info_t *last_chunk; /* The last chunk in the region (a "sentinel"). In case
                               the last chunk is allocated, the region
                               is completely full. New chunks should be inserted
                               before this chunk. */
  memory_region_t *next;
  memory_region_t *prev;
  enum allocation_strategy strategy;
  unsigned short alignment; /* alignment of the returned chunks in a 2's exponent byte count */
  ba_lock_t lock;
};

POCL_EXPORT
chunk_info_t *pocl_alloc_buffer_from_region (memory_region_t *region,
                                             size_t size);
POCL_EXPORT
chunk_info_t *pocl_alloc_buffer(memory_region_t *regions, size_t size);

POCL_EXPORT
memory_region_t *pocl_free_buffer (memory_region_t *regions, memory_address_t addr);

POCL_EXPORT
void pocl_free_chunk (chunk_info_t *chunk);

POCL_EXPORT
void pocl_init_mem_region (
    memory_region_t *region, memory_address_t start, size_t size);

chunk_info_t *create_sub_chunk (chunk_info_t *parent, size_t offset, size_t size);

void
print_chunk (chunk_info_t *chunk);

void
print_chunks (chunk_info_t *first);

#ifdef __cplusplus
}
#endif

#endif
