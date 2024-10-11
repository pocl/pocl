#define _GNU_SOURCE

#ifndef __linux__
#error This only compiles under Linux
#endif

#include <assert.h>
#include <ctype.h>
#include <string.h>

#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <ucontext.h>
#include <unistd.h>

#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_llvm.h"
#include "pocl_util.h"

static struct sigaction sigusr2_action, old_sigusr2_action;

#define FORMATTED_ULONG_MAX_LEN 20
/* formats a number to a buffer, like snprintf which can't be called from
 * signal handler returs an offset into "out" where the number starts */
static unsigned
format_int (unsigned long num, char *out)
{
  out[FORMATTED_ULONG_MAX_LEN] = 0x0A;
  unsigned i = 0;
  for (; i < FORMATTED_ULONG_MAX_LEN; ++i)
    {
      unsigned dig = num % 10;
      num = num / 10;
      out[FORMATTED_ULONG_MAX_LEN - 1 - i] = 48 + dig;
      if (num == 0)
        break;
    }
  return (FORMATTED_ULONG_MAX_LEN - 1 - i);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

#define SIGNAL_HANDLER_PRINTF(NUMBER, HEADER)                                 \
  {                                                                           \
    char store[FORMATTED_ULONG_MAX_LEN + 1];                                  \
    unsigned offset;                                                          \
    write (STDERR_FILENO, HEADER, strlen (HEADER));                           \
    offset = format_int (NUMBER, store);                                      \
    write (STDERR_FILENO, store + offset,                                     \
           (FORMATTED_ULONG_MAX_LEN + 1 - offset));                           \
  }

static void
sigusr2_signal_handler (int signo, siginfo_t *si, void *data)
{
  if (signo == SIGUSR2)
    {
      SIGNAL_HANDLER_PRINTF (context_c, "Contexts: \n");
      SIGNAL_HANDLER_PRINTF (queue_c, "Queues: \n");
      SIGNAL_HANDLER_PRINTF (buffer_c, "Buffers: \n");
      SIGNAL_HANDLER_PRINTF (svm_buffer_c, "SVM Buffers: \n");
      SIGNAL_HANDLER_PRINTF (usm_buffer_c, "USM Buffers: \n");
      SIGNAL_HANDLER_PRINTF (image_c, "Images: \n");
      SIGNAL_HANDLER_PRINTF (program_c, "Programs: \n");
      SIGNAL_HANDLER_PRINTF (kernel_c, "Kernels: \n");
      SIGNAL_HANDLER_PRINTF (sampler_c, "Samplers \n");
      SIGNAL_HANDLER_PRINTF (uevent_c, "UserEvents: \n");
      SIGNAL_HANDLER_PRINTF (event_c, "Events: \n");
    }
  else
    (*old_sigusr2_action.sa_sigaction) (signo, si, data);
}
#pragma GCC diagnostic pop

void
pocl_install_sigusr2_handler ()
{
  POCL_MSG_PRINT_GENERAL ("Installing SIGUSR2 handler...\n");

  sigusr2_action.sa_flags = SA_RESTART | SA_SIGINFO;
  sigusr2_action.sa_sigaction = sigusr2_signal_handler;
  int res = sigaction (SIGUSR2, &sigusr2_action, &old_sigusr2_action);
  assert (res == 0);
}

/* This ugly hack is required because:
 *
 * OpenCL 1.2 specification, 6.3 Operators :
 *
 * A divide by zero with integer types does not cause an exception
 * but will result in an unspecified value. Division by zero for
 * floating-point types will result in ±infinity or NaN as
 * prescribed by the IEEE-754 standard.
 *
 * FPU exceptions are masked by default on x86 linux, but integer divide
 * is not and there doesn't seem any sane way to mask it.
 *
 * This *might* be possible to fix with a LLVM pass (either check divisor
 * for 0, or perhaps some vector extension has a suitable instruction), but
 * it's likely to ruin the performance.
 */

#ifdef __x86_64__

#define DIV_OPCODE_SIZE 1
#define DIV_OPCODE_MASK 0xf6

/* F6 /6, F6 /7, F7 /6, F7 /7 */
#define DIV_OPCODE_1 0xf6
#define DIV_OPCODE_2 0xf7
#define DIV_MODRM_OPCODE_EXT_1 0x38 //  /7
#define DIV_MODRM_OPCODE_EXT_2 0x30 //  /6

#define MODRM_SIZE 1
#define MODRM_MASK 0xC0
#define REG2_MASK 0x38
#define REG1_MASK 0x07
#define ADDR_MODE_INDIRECT_ONE_BYTE_OFFSET 0x40
#define ADDR_MODE_INDIRECT_FOUR_BYTE_OFFSET 0x80
#define ADDR_MODE_INDIRECT 0x0
#define ADDR_MODE_REGISTER_ONLY 0xC0
#define REG_SP 0x4
#define REG_BP 0x5
#define SIB_BYTE 1
#define IP_RELATIVE_INDEXING 4

static struct sigaction sigfpe_action, old_sigfpe_action;

/* list of threads (e.g. of CPU driver), for which the SIGFPE should be
 * ignored. for all threads not on this list, the original handler is invoked
 */
static pocl_thread_t ignored_thread_ids[2048];
static unsigned num_ignored_threads = 0;

void
pocl_ignore_sigfpe_for_thread (pocl_thread_t thr)
{
  unsigned current_idx
      = __atomic_fetch_add (&num_ignored_threads, 1, __ATOMIC_SEQ_CST);
  __atomic_store_n (ignored_thread_ids + current_idx, thr, __ATOMIC_SEQ_CST);
}

static void
sigfpe_signal_handler (int signo, siginfo_t *si, void *data)
{
  ucontext_t *uc;
  uc = (ucontext_t *)data;
  unsigned char *eip = (unsigned char *)(uc->uc_mcontext.gregs[REG_RIP]);

  if ((signo == SIGFPE)
      && ((si->si_code == FPE_INTDIV) || (si->si_code == FPE_INTOVF)))
    {
      /* SIGFPE is delivered to the thread that caused the div-by-zero.
       * check if the thread is on the list of threads we should ignore.
       */
      pocl_thread_t ID = POCL_THREAD_SELF();
      int found = 0;
      unsigned max_threads
          = __atomic_load_n (&num_ignored_threads, __ATOMIC_SEQ_CST);
      for (unsigned i = 0; i < max_threads; ++i)
        {
          if (ID == ignored_thread_ids[i])
            {
              found = 1;
              break;
            }
        }
      /* if it's not on the list, run the original handler. */
      if (!found)
        goto ORIGINAL_HANDLER;

      /* Luckily for us, div-by-0 exceptions do NOT advance the IP register,
       * so we have to disassemble the instruction (to know its length)
       * and move IP past it. */
      unsigned n = 0;

      /* skip all prefixes */
      while ((n < 4) && ((eip[n] & DIV_OPCODE_MASK) != DIV_OPCODE_MASK))
        ++n;

      /* too much prefixes = decoding failed */
      if (n >= 4)
        goto ORIGINAL_HANDLER;

      /* check opcode */
      unsigned opcode = eip[n];
      if ((opcode != DIV_OPCODE_1) && (opcode != DIV_OPCODE_2))
        goto ORIGINAL_HANDLER;
      n += DIV_OPCODE_SIZE;

      unsigned modrm = eip[n];
      unsigned modmask = modrm & MODRM_MASK;
      unsigned reg1mask = modrm & REG1_MASK;
      unsigned reg2mask = modrm & REG2_MASK;
      /* check opcode extension in ModR/M reg2 */
      if ((reg2mask != DIV_MODRM_OPCODE_EXT_1)
          && (reg2mask != DIV_MODRM_OPCODE_EXT_2))
        goto ORIGINAL_HANDLER;
      n += MODRM_SIZE;

      /* handle immediates/registers */
      if (modmask == ADDR_MODE_INDIRECT_ONE_BYTE_OFFSET)
        n += 1;
      if (modmask == ADDR_MODE_INDIRECT_FOUR_BYTE_OFFSET)
        n += 4;
      if (modmask == ADDR_MODE_INDIRECT)
        n += 0;
      if (modmask != ADDR_MODE_REGISTER_ONLY)
        {
          if (reg1mask == REG_SP)
            n += SIB_BYTE;
          if (reg1mask == REG_BP)
            n += IP_RELATIVE_INDEXING;
        }

      uc->uc_mcontext.gregs[REG_RIP] += n;
      return;
    }
  else
    {
    ORIGINAL_HANDLER:
      (*old_sigfpe_action.sa_sigaction) (signo, si, data);
    }
}

#endif

static char signal_empty_file[POCL_MAX_PATHNAME_LENGTH];

void
pocl_install_sigfpe_handler ()
{
#ifdef __x86_64__

#ifdef ENABLE_LLVM
  /* This is required to force LLVM to register its signal
   * handlers, before pocl registers its own SIGFPE handler.
   * LLVM otherwise calls this via
   *    pocl_llvm_build_program ->
   *    clang::PrintPreprocessedAction ->
   *    CreateOutputFile -> RemoveFileOnSignal
   * Registering our handlers before LLVM creates its sigaltstack
   * leads to interesting crashes & bugs later.
   */
  pocl_cache_tempname (signal_empty_file, NULL, NULL);
  pocl_llvm_remove_file_on_signal_create (signal_empty_file);
#endif

  POCL_MSG_PRINT_GENERAL ("Installing SIGFPE handler...\n");

  sigfpe_action.sa_flags = SA_RESTART | SA_SIGINFO;
  sigfpe_action.sa_sigaction = sigfpe_signal_handler;
  int res = sigaction (SIGFPE, &sigfpe_action, &old_sigfpe_action);
  assert (res == 0);
#endif
}

void
pocl_destroy_sigfpe_handler ()
{
#ifdef ENABLE_LLVM
  pocl_llvm_remove_file_on_signal_destroy (signal_empty_file);
#endif
}
