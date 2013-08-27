/* OpenCL runtime library: pocl_util utility functions

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pocl_util.h"
#include "pocl_cl.h"
#include "utlist.h"

#define TEMP_DIR_PATH_CHARS 16

void 
remove_directory (const char *path_name) 
{
  int str_size = 8 + strlen(path_name) + 1;
  char *cmd = (char*)malloc(str_size);
  snprintf (cmd, str_size, "rm -fr %s", path_name);
  system (cmd);
  free (cmd);
}

#define POCL_TEMPDIR_ENV "POCL_TEMP_DIR"

char*
pocl_create_temp_dir() 
{  
  struct temp_dir *td; 
  char *path_name; 
  if (getenv(POCL_TEMPDIR_ENV) != NULL &&
      access (getenv(POCL_TEMPDIR_ENV), F_OK) == 0) 
    {
      path_name = (char*)malloc (strlen(getenv(POCL_TEMPDIR_ENV)) + 1);
      strcpy (path_name, getenv(POCL_TEMPDIR_ENV));
    }
  else 
    {
      path_name = (char*)malloc (TEMP_DIR_PATH_CHARS);
      strncpy (path_name, "/tmp/poclXXXXXX\0", TEMP_DIR_PATH_CHARS);
      mkdtemp (path_name);  
    }
  return path_name;
}

uint32_t
byteswap_uint32_t (uint32_t word, char should_swap) 
{
    union word_union 
    {
        uint32_t full_word;
        unsigned char bytes[4];
    } old, neww;
    if (!should_swap) return word;

    old.full_word = word;
    neww.bytes[0] = old.bytes[3];
    neww.bytes[1] = old.bytes[2];
    neww.bytes[2] = old.bytes[1];
    neww.bytes[3] = old.bytes[0];
    return neww.full_word;
}

float
byteswap_float (float word, char should_swap) 
{
    union word_union 
    {
        float full_word;
        unsigned char bytes[4];
    } old, neww;
    if (!should_swap) return word;

    old.full_word = word;
    neww.bytes[0] = old.bytes[3];
    neww.bytes[1] = old.bytes[2];
    neww.bytes[2] = old.bytes[1];
    neww.bytes[3] = old.bytes[0];
    return neww.full_word;
}

size_t
pocl_size_ceil2(size_t x) {
  /* Rounds up to the next highest power of two without branching and
   * is as fast as a BSR instruction on x86, see:
   *
   * http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
   */
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFF
  x |= x >> 32;
#endif
  return ++x;
}

#ifndef HAVE_ALIGNED_ALLOC
void *
pocl_aligned_malloc(size_t alignment, size_t size)
{
# ifdef HAVE_POSIX_MEMALIGN
  
  /* make sure that size is a multiple of alignment, as posix_memalign
   * does not perform this test, whereas aligned_alloc does */
  if ((size & (alignment - 1)) != 0)
    {
      errno = EINVAL;
      return NULL;
    }

  /* posix_memalign requires alignment to be at least sizeof(void *) */
  if (alignment < sizeof(void *))
    alignment = sizeof(void* );

  void* result;
  int err;
  
  err = posix_memalign(&result, alignment, size);
  if (err)
    {
      errno = err;
      return NULL;
    }

  return result;

# else
  
  /* allow zero-sized allocations, force alignment to 1 */
  if (!size)
    alignment = 1;

  /* make sure alignment is a non-zero power of two and that
   * size is a multiple of alignment */
  size_t mask = alignment - 1;
  if (!alignment || ((alignment & mask) != 0) || ((size & mask) != 0))
    {
      errno = EINVAL;
      return NULL;
    }

  /* allocate memory plus space for alignment header */
  uintptr_t address = (uintptr_t)malloc(size + mask + sizeof(void *));
  if (!address)
    return NULL;

  /* align the address, and store original pointer for future use
   * with free in the preceeding bytes */
  uintptr_t aligned_address = (address + mask + sizeof(void *)) & ~mask;
  void** address_ptr = (void **)(aligned_address - sizeof(void *));
  *address_ptr = (void *)address;
  return (void *)aligned_address;

#endif
}
#endif

#if !defined HAVE_ALIGNED_ALLOC && !defined HAVE_POSIX_MEMALIGN
void
pocl_aligned_free(void *ptr)
{
  /* extract pointer from original allocation and free it */
  if (ptr)
    free(*(void **)((uintptr_t)ptr - sizeof(void *)));
}


cl_int pocl_create_event (cl_event *event, cl_command_queue command_queue, 
                          cl_command_type command_type)
{
  if (event != NULL)
    {
      *event = (cl_event)malloc (sizeof (struct _cl_event));
      if (event == NULL)
        return CL_OUT_OF_HOST_MEMORY;
      
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      POname(clRetainCommandQueue) (command_queue);
      (*event)->command_type = command_type;
    }
  return CL_SUCCESS;
}

cl_int pocl_create_command (_cl_command_node **cmd, 
                            cl_command_queue command_queue, 
                            cl_command_type command_type, cl_event *event, 
                            cl_int num_events, const cl_event *wait_list)
{
  int i;
  
  if ((wait_list == NULL && num_events != 0) ||
      (wait_list != NULL && num_events == 0))
    return CL_INVALID_EVENT_WAIT_LIST;
  
  for (i = 0; i < num_events; ++i)
    {
      if (wait_list[i] == NULL)
        return CL_INVALID_EVENT_WAIT_LIST;
    }
  
  (*cmd) = (_cl_command_node*)malloc (sizeof (_cl_command_node));
  if (*cmd == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  
  (*cmd)->type = command_type;
  (*cmd)->next = NULL;
  (*cmd)->event = event ? (*event) : NULL;
  (*cmd)->event_wait_list = wait_list;
  (*cmd)->num_events_in_wait_list = num_events;
  (*cmd)->device = command_queue->device;

  return CL_SUCCESS;
}
#endif

