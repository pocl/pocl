/* OpenCL runtime/device driver library: custom buffer allocator

   Copyright (c) 2011-2020 pocl contributors

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
 * The allocator is designed to be fast and simple by tuning it to the
 * common OpenCL allocation patterns. It can be used from host to allocate
 * external device's memory or it can be used from within the device to
 * dynamically allocate local memory chunks. The interface is address space
 * agnostic; it treats memory addresses and regions of memory as integers.
 *
 * Certain assumptions of OpenCL allocation patterns are made to optimize and
 * simplify the implementation:
 *
 * 1) The allocations are often quite large and the "lifetimes" of the
 * buffers are of similar length.
 *
 * This affects memory fragmentation patterns. When two or more buffers
 * with similar life times are allocated (typically all the buffer
 * arguments to an OpenCL kernel invocation) and freed we should try to
 * allocate sequential chunks for the buffers and in the free merge
 * the freed buffer with a succeeding and preceeding ones. This should
 * lead to large physical chunks allocated for each kernel's buffers
 * which are also deallocated back to large region chunks.
 *
 * 2) The total number of allocations is not large.
 *
 * Traversing through the list of chunk infos when searching for an
 * available freed chunk can be considered to be not very costly.
 *
 * 3a) There is no lack of (global) memory.
 *
 * A wasteful but fast strategy can be used. Here the chunk is always
 * tried to be allocated to the end of the region to enforce "sequential
 * large chunks" for multiple buffers of a kernel.
 *
 * 3b) The memory is more tight (e.g. when allocating from local memory).
 *
 * A slower but less wasteful strategy should be used. In this version
 * the list of old chunks should be traversed first and reused in case
 * a large enough unallocated one is found. This version can be used
 * also for the case where there's a single region (basically heap) that
 * grows towards the stack or the global data area of the memory.
 *
 * @author Pekka Jääskeläinen 2011-2012
 *
 * @file bufalloc.c
 */

#include "bufalloc.h"
#include "utlist.h"
#include "pocl_icd.h"

//#define DEBUG_BUFALLOC


#include <stdio.h>

void
print_chunk (chunk_info_t *chunk)
{
  printf ("### chunk %p: allocated: %d start: %zx size: %zu prev: %p next: %p\n",
          chunk, chunk->is_allocated, chunk->start_address,
          chunk->size, chunk->prev, chunk->next);
}

void
print_chunks (chunk_info_t *first)
{
  chunk_info_t *chunk;
  DL_FOREACH (first, chunk)
    {
      print_chunk (chunk);
    }
}

static int
chunk_slack (chunk_info_t* chunk, size_t size, size_t* last_chunk_size)
{
  memory_address_t aligned_start_addr =
    (chunk->start_address + chunk->parent_region->alignment - 1) &
    ~(chunk->parent_region->alignment - 1);
  size_t end_chunk = chunk->start_address + chunk->size;
  size_t end_addr = aligned_start_addr + size;
  if(last_chunk_size)
    *last_chunk_size = end_chunk - end_addr;
  return end_chunk >= end_addr;
}

/**
 * Tries to create a new chunk to the end of the given region.
 *
 * Assumes the last_chunk is always the last chunk in the region and
 * it is unallocated in case there is free space at the end.
 *
 * @return The address of the chunk if it fits, 0 otherwise.
 */
static chunk_info_t *
append_new_chunk (memory_region_t *region,
                  size_t size)
{

  chunk_info_t* new_chunk = NULL;
  BA_LOCK (region->lock);

#ifdef ENABLE_ASSERTS
  assert (!region->last_chunk->is_allocated);
#endif
  /* if the last_chunk is too small we cannot append
     a new chunk before it */
  if (!chunk_slack (region->last_chunk, size, NULL))
    {
      BA_UNLOCK (region->lock);
      return NULL;
    }

  /* ok, there should be space at the end, create a new chunk
     before the last_chunk */
  new_chunk = region->free_chunks;

  if (new_chunk == NULL)
    {
      BA_UNLOCK (region->lock);
      return NULL;
    }
  else
    {
      DL_DELETE (region->free_chunks, new_chunk);
    }

  /* Round the start address up towards the closest aligned
     address. */
  new_chunk->start_address =
    (region->last_chunk->start_address + region->alignment - 1) &
    ~(region->alignment - 1);
  new_chunk->parent_region = region;
  new_chunk->size = size;
  new_chunk->is_allocated = 1;
  new_chunk->children = NULL;

  chunk_slack (region->last_chunk, size, (size_t*)&region->last_chunk->size);
  region->last_chunk->start_address =
    new_chunk->start_address + new_chunk->size;

  DL_DELETE (region->chunks, region->last_chunk);
  DL_APPEND (region->chunks, new_chunk);
  DL_APPEND (region->chunks, region->last_chunk);

#ifdef DEBUG_BUFALLOC
  printf ("#### after append_new_chunk (%x, %u)\n", region, size);
  print_chunks (region->chunks);
  printf ("\n");
#endif

  BA_UNLOCK (region->lock);

  return new_chunk;
}

/**
 * Allocates a chunk of memory from the given memory region.
 *
 * @return The chunk, or NULL if no space available in the region.
 */
chunk_info_t*
alloc_buffer_from_region (memory_region_t *region, size_t size)
{
#ifdef ENABLE_ASSERTS
  assert (region != NULL);
#endif
  /* The memory-wasteful but fast strategy:

     Assume there's plenty of memory so just try to append the
     buffer to the end of the region without trying to reuse
     unallocated ones first. */
  chunk_info_t* chunk = NULL, *cursor;
  if (region->strategy == BALLOCS_WASTEFUL)
    {
      chunk = append_new_chunk(region, size);
      if (chunk != NULL) return chunk;
    }

  BA_LOCK (region->lock);

  DL_FOREACH (region->chunks, cursor)
    {
      if (cursor == region->last_chunk ||
          cursor->is_allocated ||
          !chunk_slack (cursor, size, NULL))
        {
#ifdef DEBUG_BUFALLOC
          printf ("### CHUNK REJECTED: SIZE: %zu IS_LAST: %u CHUNK %p: "
                  "allocated: %d start: %zx size: %zu prev: %p next: %p\n",
          size, (unsigned)(cursor == region->last_chunk),
          cursor, cursor->is_allocated, cursor->start_address,
          cursor->size, cursor->prev, cursor->next);
#endif
          continue; /* doesn't fit */
        }
      /* found one */
      chunk = cursor;
      chunk->is_allocated = 1;

#ifdef DEBUG_BUFALLOC
      printf ("#### after reusing a chunk in region %x\n", region);
      print_chunks (region->chunks);
      printf ("\n");
#endif
      break;
    }

  BA_UNLOCK (region->lock);

  if (chunk == NULL && region->strategy != BALLOCS_WASTEFUL)
    {
      return append_new_chunk (region, size);
    }
  return chunk;
}

/**
 * Allocates a chunk of memory from one of the given memory regions.
 *
 * The address ranges of the different regions must not overlap. Searches
 * through the regions in the order of the region pointer array.
 *
 * @param regions A linked list of region pointers.
 * @param size The size of the chunk to allocate.
 * @return The start address of the chunk, or 0 if no space available
 * in the buffer.
 */

#ifndef BUFALLOC_NO_MULTIPLE_REGIONS 
chunk_info_t *
alloc_buffer (memory_region_t *regions, size_t size)
{
  chunk_info_t *chunk = NULL;
  memory_region_t *region = NULL;
  LL_FOREACH(regions, region)
    {
      chunk = alloc_buffer_from_region (region, size);
      if (chunk != NULL)
        return chunk;
    }
  return NULL;
}
#endif

/**
 * Creates a reference to a part of a chunk.
 *
 * @todo Register to the parent also so it can free the
 * child references.
 */
#ifndef BUFALLOC_NO_SUB_CHUNKS
chunk_info_t *
create_sub_chunk (chunk_info_t *parent, size_t offset, size_t size)
{
  chunk_info_t *subchunk = (chunk_info_t*)malloc(sizeof(struct chunk_info));
  subchunk->start_address = parent->start_address + offset;
  subchunk->size = size;
  subchunk->parent = parent;
  DL_APPEND(parent->children, subchunk);
  return subchunk;
}
#endif

/**
 * Merges two unallocated chunks together to a larger chunk, if both
 * are unallocated.
 *
 * The first chunk must before the second chunk in the memory region.
 * Must be called inside a locked region.
 *
 * @return A pointer to the coalesced chunk, or the second chunk in case
 * coalsecing could not be done.
 */

#ifndef BUFALLOC_NO_CHUNK_COALESCING
static chunk_info_t * 
coalesce_chunks (chunk_info_t* first, 
                 chunk_info_t* second)
{
  if (first == NULL) return second;
  if (second == NULL) return first;
  if (first->is_allocated || second->is_allocated) return second;

  /* The linked list head has a prev pointing to the last (sentinel),
     detect that here and do not merge first with the second. */
  if (first->start_address > second->start_address) return second;

#ifdef DEBUG_BUFALLOC
  printf ("### coalescing chunks:\n");
  print_chunk (first);
  print_chunk (second);
  puts ("\n");
#endif

  /* Should not just add the size of the second chunk as we might have
     done alignment adjustment to the start address */
  first->size = second->start_address + second->size - first->start_address;
  DL_DELETE (first->parent_region->chunks, second);
  DL_APPEND (first->parent_region->free_chunks, second);

  /* Did we coalesce away the sentinel chunk? Need to set it to
     a new valid one. */
  if (second->parent_region->last_chunk == second)
      second->parent_region->last_chunk = first;

  return first;
}
#endif

memory_region_t *
free_buffer (memory_region_t *regions, memory_address_t addr)
{
  memory_region_t *region = NULL;

#ifdef DEBUG_BUFALLOC
  printf ("#### free_buffer(%p, %x)\n", regions, addr);
#endif

  LL_FOREACH (regions, region)
    {
      chunk_info_t *chunk = NULL;
      BA_LOCK (region->lock);
      DL_FOREACH (region->chunks, chunk)
        {
          if (chunk->start_address == addr)
            {
              chunk->is_allocated = 0;
#ifndef BUFALLOC_NO_CHUNK_COALESCING
              coalesce_chunks (coalesce_chunks (chunk->prev, chunk), chunk->next);
#endif
              BA_UNLOCK (region->lock);
#ifdef DEBUG_BUFALLOC
              printf ("#### region %x after free_buffer at addr %x\n",
                      region, addr);
              print_chunks (region->chunks);
              printf ("\n");
#endif
              return region;
            }
        }
      BA_UNLOCK (region->lock);
    }
  return NULL;
}

/**
 * Frees the given chunk.
 *
 * Successive unallocated chunks in the region, if found, are merged to
 * form larger unallocated chunks.
 */
void
free_chunk (chunk_info_t* chunk)
{
  memory_region_t *region = chunk->parent_region;
  BA_LOCK (region->lock);
  chunk->is_allocated = 0;
#ifndef BUFALLOC_NO_CHUNK_COALESCING
  coalesce_chunks (coalesce_chunks (chunk->prev, chunk), chunk->next);
#endif
  BA_UNLOCK (region->lock);

#ifdef DEBUG_BUFALLOC
  printf ("#### after free_chunk (%x)\n", chunk);
  print_chunks (region->chunks);
  printf ("\n");
#endif

}

/** Initialize a memory_region_t.
 * @param region is a pointer to a existing memory_region_t data structure.
 * @param start the base address of the memory region to be managed.
 * @Param size  the size of the region (in bytes?)
 */
void
init_mem_region (memory_region_t *region, memory_address_t start, size_t size)
{
  int i;
  BA_INIT_LOCK (region->lock);

  region->strategy = BALLOCS_WASTEFUL;
  region->chunks = NULL;
  region->free_chunks = NULL;
  region->alignment = 64;
  region->next = NULL;
  region->prev = NULL;
  /* Create the "sentinel chunk" */
  region->last_chunk = &region->all_chunks[0];
  region->last_chunk->start_address = start;
  region->last_chunk->size = size;
  region->last_chunk->is_allocated = 0;
  region->last_chunk->parent_region = region;

  DL_APPEND(region->chunks, region->last_chunk);

  /* Setup the linked list of free chunk data structures */
  for (i = 1; i < MAX_CHUNKS_IN_REGION; ++i)
    DL_APPEND (region->free_chunks, &region->all_chunks[i]);

#ifdef DEBUG_BUFALLOC
  printf ("#### memory region %x created. start: %x size: %u\n",
          region, start, size);
#endif
}
