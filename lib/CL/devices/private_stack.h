/* private_stack.h - running host code on a stack owned by PoCL

   Copyright (c) 2026 PoCL developers

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

/* Kernels executed on an application thread can't use that thread's stack:
   how much of it is left is unknown, and when the application runs its own
   coroutines (e.g. Julia tasks) the stack PoCL is called on isn't even the
   thread's stack. Instead, such code runs on a separate stack of a known size,
   on the same thread, through a small per-architecture trampoline. */

#ifndef POCL_PRIVATE_STACK_H
#define POCL_PRIVATE_STACK_H

#include <stddef.h>

#include "pocl_export.h"

/* Not on arm64e, where function pointers are signed and the trampoline would
   have to authenticate the call. */
#if !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))             \
  && (defined(__x86_64__) || (defined(__aarch64__) && !defined(__arm64e__)))
#define POCL_HAVE_PRIVATE_STACK 1
#endif

#ifdef POCL_HAVE_PRIVATE_STACK

typedef struct pocl_private_stack
{
  /* The mapping, including the guard region at its low end. */
  void *base;
  size_t size;
  /* The usable range [lo, lo + usable_size). */
  void *lo;
  size_t usable_size;
  unsigned valgrind_id;
} pocl_private_stack;

/* Maps a stack with SIZE usable bytes above an inaccessible guard region.
   Returns 0 on success. */
POCL_EXPORT
int pocl_private_stack_alloc (pocl_private_stack *stack, size_t size);

POCL_EXPORT
void pocl_private_stack_free (pocl_private_stack *stack);

/* Calls FN (ARG) on STACK, on the calling thread, and returns once FN does.
   FN must return normally: unwinding or longjmp out of it is not supported. */
POCL_EXPORT
void pocl_run_on_private_stack (pocl_private_stack *stack,
                                void (*fn) (void *),
                                void *arg);

#endif

#endif
