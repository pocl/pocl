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


