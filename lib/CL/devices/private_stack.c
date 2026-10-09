/* private_stack.c - running host code on a stack owned by PoCL

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

#include "config.h"

#include "private_stack.h"

#ifdef POCL_HAVE_PRIVATE_STACK

#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef ENABLE_VALGRIND
#include <valgrind/valgrind.h>
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define POCL_ASAN_STACK_SWITCH 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define POCL_ASAN_STACK_SWITCH 1
#endif
#ifdef POCL_ASAN_STACK_SWITCH
#include <sanitizer/common_interface_defs.h>
#endif

/* Catches overflows by at least this much; frames larger than this can skip
   over it, as with any guard page. */
#define GUARD_SIZE (64 * 1024)

/* Calls FN (ARG) with the stack pointer set to STACK_TOP, then restores it.

   The frame pointer saved on entry also serves as the canonical frame address,
   so that debuggers and profilers unwind from FN back into the caller. */
void pocl_private_stack_call (void *arg, void (*fn) (void *), void *stack_top)
  __attribute__ ((visibility ("hidden")));

#ifdef __APPLE__
#define FUNC_BEGIN                                                            \
  ".text\n"                                                                   \
  ".globl _pocl_private_stack_call\n"                                         \
  ".private_extern _pocl_private_stack_call\n"                                \
  ".p2align 4\n"                                                              \
  "_pocl_private_stack_call:\n"
#define FUNC_END ""
#else
#define FUNC_BEGIN                                                            \
  ".pushsection .text\n"                                                      \
  ".globl pocl_private_stack_call\n"                                          \
  ".hidden pocl_private_stack_call\n"                                         \
  ".type pocl_private_stack_call, %function\n"                                \
  ".p2align 4\n"                                                              \
  "pocl_private_stack_call:\n"
#define FUNC_END                                                              \
  ".size pocl_private_stack_call, .-pocl_private_stack_call\n"                \
  ".popsection\n"
#endif

#if defined(__x86_64__)
#ifdef __CET__
#define LANDING_PAD "endbr64\n"
#else
#define LANDING_PAD ""
#endif
/* clang-format off */
__asm__ (FUNC_BEGIN
         ".cfi_startproc\n"
         LANDING_PAD
         "pushq %rbp\n"
         ".cfi_def_cfa_offset 16\n"
         ".cfi_offset %rbp, -16\n"
         "movq %rsp, %rbp\n"
         ".cfi_def_cfa_register %rbp\n"
         "movq %rdx, %rsp\n"
         "callq *%rsi\n"
         "movq %rbp, %rsp\n"
         "popq %rbp\n"
         ".cfi_def_cfa %rsp, 8\n"
         ".cfi_restore %rbp\n"
         "retq\n"
         ".cfi_endproc\n"
         FUNC_END);
/* clang-format on */
#elif defined(__aarch64__)
/* "hint #34" is "bti c", a no-op on cores without BTI. */
/* clang-format off */
__asm__ (FUNC_BEGIN
         ".cfi_startproc\n"
         "hint #34\n"
         "stp x29, x30, [sp, #-16]!\n"
         ".cfi_def_cfa_offset 16\n"
         ".cfi_offset x30, -8\n"
         ".cfi_offset x29, -16\n"
         "mov x29, sp\n"
         ".cfi_def_cfa_register x29\n"
         "mov sp, x2\n"
         "blr x1\n"
         "mov sp, x29\n"
         ".cfi_def_cfa_register sp\n"
         "ldp x29, x30, [sp], #16\n"
         ".cfi_def_cfa_offset 0\n"
         ".cfi_restore x30\n"
         ".cfi_restore x29\n"
         "ret\n"
         ".cfi_endproc\n"
         FUNC_END);
/* clang-format on */
#else
#error "POCL_HAVE_PRIVATE_STACK is defined for an unsupported architecture"
#endif

int
pocl_private_stack_alloc (pocl_private_stack *stack, size_t size)
{
  size_t page = (size_t)sysconf (_SC_PAGESIZE);
  size_t guard = (GUARD_SIZE + page - 1) & ~(page - 1);
  size = (size + page - 1) & ~(page - 1);

  int flags = MAP_PRIVATE | MAP_ANON;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
#if defined(__linux__) && defined(MAP_STACK)
  flags |= MAP_STACK;
#endif
  void *base = mmap (NULL, guard + size, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (base == MAP_FAILED)
    return -1;
  if (mprotect (base, guard, PROT_NONE) != 0)
    {
      munmap (base, guard + size);
      return -1;
    }

  stack->base = base;
  stack->size = guard + size;
  stack->lo = (char *)base + guard;
  stack->usable_size = size;
#ifdef ENABLE_VALGRIND
  stack->valgrind_id = VALGRIND_STACK_REGISTER (
    stack->lo, (char *)stack->lo + stack->usable_size);
#endif
  return 0;
}

void
pocl_private_stack_free (pocl_private_stack *stack)
{
  if (stack->base == NULL)
    return;
#ifdef ENABLE_VALGRIND
  VALGRIND_STACK_DEREGISTER (stack->valgrind_id);
#endif
  munmap (stack->base, stack->size);
  stack->base = NULL;
}

#ifdef POCL_ASAN_STACK_SWITCH
/* AddressSanitizer has to be told about the switches, or it misreports
   accesses to the stack that isn't current. */
struct asan_call
{
  void (*fn) (void *);
  void *arg;
  const void *caller_bottom;
  size_t caller_size;
};

static void
asan_call_on_stack (void *p)
{
  struct asan_call *call = (struct asan_call *)p;
  __sanitizer_finish_switch_fiber (NULL, &call->caller_bottom,
                                   &call->caller_size);
  call->fn (call->arg);
  __sanitizer_start_switch_fiber (NULL, call->caller_bottom,
                                  call->caller_size);
}
#endif

void
pocl_run_on_private_stack (pocl_private_stack *stack,
                           void (*fn) (void *),
                           void *arg)
{
  void *top
    = (void *)(((uintptr_t)stack->lo + stack->usable_size) & ~(uintptr_t)15);
#ifdef POCL_ASAN_STACK_SWITCH
  struct asan_call call = { fn, arg, NULL, 0 };
  void *fake_stack = NULL;
  __sanitizer_start_switch_fiber (&fake_stack, stack->lo, stack->usable_size);
  pocl_private_stack_call (&call, asan_call_on_stack, top);
  __sanitizer_finish_switch_fiber (fake_stack, NULL, NULL);
#else
  pocl_private_stack_call (arg, fn, top);
#endif
}

#endif
