/* OpenCL runtime library: command running utility functions

   Copyright (c) 2012-2019 Pekka Jääskeläinen
                 2020-2024 PoCL Developers
                 2024 Pekka Jääskeläinen / Intel Finland Oy
                 2024 Henry Linjamäki / Intel Finland Oy

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

#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#ifdef __MINGW32__
#include <process.h>
#endif

#ifndef _WIN32
#include <dirent.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utime.h>
#else
#include "vccompat.hpp"
#endif

#include "pocl_cl.h"
#include "pocl_compiler_macros.h"
#include "pocl_run_command.h"

/** Builds command string from tokenized arguments.
 *
 * \return New allocation holding the command string. NULL if an error
 * is encountered.
 */
static char *
build_cmd_from_arglist (const char **args)
{
  /* Calculate required buffer size for all arguments. */
  size_t total_len = 0;
  for (const char **arg = args; *arg != NULL; arg++)
    {
      /* Account for spaces, quotes, and worst-case escaping. */
      total_len += strlen (*arg) * 2 + 3;
    }

  char *cmd = (char *)malloc (total_len);
  if (!cmd)
    return NULL;
  cmd[0] = '\0';

  /* Build command line with proper escaping. */
  for (const char **arg = args; *arg != NULL; arg++)
    {
      /* Add space between arguments. */
      if (arg != args)
        strcat (cmd, " ");

      /* Check if we need quotes (contains space or empty). */
      int needs_quotes = (strchr (*arg, ' ') != NULL) || (*arg[0] == '\0');
      if (needs_quotes)
        strcat (cmd, "\"");

      /* Copy and escape argument. */
      char *dst = cmd + strlen (cmd);
      for (const char *src = *arg; *src != '\0'; src++)
        {
          if (*src == '"')
            *dst++ = '\\';
          *dst++ = *src;
        }
      *dst = '\0';
      if (needs_quotes)
        strcat (cmd, "\"");
    }

  return cmd;
}

int
pocl_run_command (const char **args)
{
  POCL_MSG_PRINT_INFO ("Launching: %s\n", args[0]);
#if defined(HAVE_FORK)
  pid_t p = fork ();
  if (p == 0)
    {
      execv (args[0], (char *const *)args);
      POCL_MSG_ERR ("execv failed!");
      abort ();
    }
  else
    {
      if (p < 0)
        return EXIT_FAILURE;
      int status;
      int ret;
      do
        {
          ret = waitpid (p, &status, 0);
        }
      while (ret == -1 && errno == EINTR);
      if (ret < 0)
        {
          POCL_MSG_ERR ("pocl: waitpid() failed.\n");
          return EXIT_FAILURE;
        }
      if (WIFEXITED (status))
        return WEXITSTATUS (status);
      else if (WIFSIGNALED (status))
        return WTERMSIG (status);
      else
        return EXIT_FAILURE;
    }
#elif _WIN32
  STARTUPINFO si;
  ZeroMemory (&si, sizeof (si));
  si.cb = sizeof (si);
  PROCESS_INFORMATION pi;
  ZeroMemory (&pi, sizeof (pi));
  DWORD dwProcessFlags = 0;

  char *cmd = build_cmd_from_arglist (args);
  if (!cmd)
    return EXIT_FAILURE;

  POCL_MSG_PRINT_INFO ("Running command: %s\n", cmd);
  int success = CreateProcess (NULL, cmd, NULL, NULL, TRUE, dwProcessFlags,
                               NULL, NULL, &si, &pi);
  free (cmd);

  if (!success)
    return EXIT_FAILURE;

  DWORD waitRc = WaitForSingleObject (pi.hProcess, INFINITE);
  if (waitRc == WAIT_FAILED)
    return EXIT_FAILURE;

  DWORD exit_code = 0;
  success = GetExitCodeProcess (pi.hProcess, &exit_code);

  CloseHandle (pi.hProcess);
  CloseHandle (pi.hThread);
  if (!success)
    return EXIT_FAILURE;
  return exit_code;
#else
#error Must have fork() or vfork() or Win32 CreateProcess
#endif
}

int
pocl_run_command_capture_output (char *capture_string,
                                 size_t *captured_bytes,
                                 const char **args)
{
#if defined(HAVE_FORK)
  POCL_MSG_PRINT_INFO ("Launching: %s\n", args[0]);

  int in[2];
  int out[2];
  pipe (in);
  pipe (out);

  pid_t p = fork ();
  if (p == 0)
    {
      close (in[1]);
      close (out[0]);

      dup2 (in[0], STDIN_FILENO);
      dup2 (out[1], STDOUT_FILENO);
      dup2 (out[1], STDERR_FILENO);

      execv (args[0], (char *const *)args);
      POCL_MSG_ERR ("execv failed!");
      abort ();
    }
  else
    {
      if (p < 0)
        return EXIT_FAILURE;

      close (in[0]);
      close (out[1]);

      ssize_t r = 0;
      size_t total_bytes = 0;
      size_t capture_limit = *captured_bytes;
      char buf[4096];

      while ((r = read (out[0], buf, 4096)) > 0)
        {
          if (total_bytes + r > capture_limit)
            /* Read out the bytes even if they don't fit to the buffer to
           not block the pipe. */
            continue;
          memcpy (capture_string + total_bytes, buf, r);
          total_bytes += r;
        }
      if (total_bytes > capture_limit)
        total_bytes = capture_limit;

      capture_string[total_bytes] = 0;
      *captured_bytes = total_bytes;

      int status;
      int ret;
      do
        {
          ret = waitpid (p, &status, 0);
        }
      while (ret == -1 && errno == EINTR);
      if (ret < 0)
        {
          POCL_MSG_ERR ("pocl: waitpid() failed.\n");
          return EXIT_FAILURE;
        }

      close (out[0]);
      close (in[1]);

      if (WIFEXITED (status))
        return WEXITSTATUS (status);
      else if (WIFSIGNALED (status))
        return WTERMSIG (status);
      else
        return EXIT_FAILURE;
    }
  return EXIT_FAILURE;
#elif _WIN32 // ^ HAVE_FORK ^
  char *cmd = build_cmd_from_arglist (args);
  if (!cmd)
    return EXIT_FAILURE;

  /* Based on https://stackoverflow.com/questions/42402673/createprocess-and
   * -capture-stdout */
  HANDLE child_stdout_rd = NULL;
  HANDLE child_stdout_wr = NULL;
  STARTUPINFO startup_info;
  PROCESS_INFORMATION process_info;
  SECURITY_ATTRIBUTES sec_attrs;

  ZeroMemory (&sec_attrs, sizeof (sec_attrs));
  sec_attrs.nLength = sizeof (SECURITY_ATTRIBUTES);
  sec_attrs.bInheritHandle = TRUE;
  sec_attrs.lpSecurityDescriptor = NULL;

  // Create a pipe for the child process's STDOUT.
  if (!CreatePipe (&child_stdout_rd, &child_stdout_wr, &sec_attrs, 0))
    return EXIT_FAILURE;

  // Ensure the read handle to the pipe for STDOUT is not inherited.
  if (!SetHandleInformation (child_stdout_rd, HANDLE_FLAG_INHERIT, 0))
    {
      CloseHandle (child_stdout_rd);
      CloseHandle (child_stdout_wr);
      return EXIT_FAILURE;
    }

  ZeroMemory (&startup_info, sizeof (startup_info));
  startup_info.cb = sizeof (startup_info);
  startup_info.hStdError = child_stdout_wr;
  startup_info.hStdOutput = child_stdout_wr;
  startup_info.dwFlags |= STARTF_USESTDHANDLES;

  ZeroMemory (&process_info, sizeof (process_info));

  if (!CreateProcess (NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL,
                      &startup_info, &process_info))
    {
      CloseHandle (child_stdout_rd);
      CloseHandle (child_stdout_wr);
      return EXIT_FAILURE;
    }

  /* We are not using the the write-end of the pipe and need to close it
   * so we get ERROR_BROKEN_PIPE from ReadFile() when the command is done
   * writing into the pipe.  */
  CloseHandle (child_stdout_wr);

  DWORD bytes_to_read = *captured_bytes;
  while (1)
    {
      DWORD bytes_read = 0;
      BOOL success = ReadFile (child_stdout_rd, capture_string, bytes_to_read,
                               &bytes_read, NULL);
      if (!success)
        {
          if (GetLastError () == ERROR_BROKEN_PIPE)
            break; /* No more data from the child process.  */

          assert (!"Unhandled error from pipe.");
          break;
        }
      if (!bytes_read)
        continue;

      assert (bytes_to_read >= bytes_read);
      bytes_to_read -= bytes_read;
      capture_string += bytes_read;
      if (bytes_to_read == 0)
        break; /* Caller's buffer is full.  */
    }
  *captured_bytes -= bytes_to_read;

  /* In case the command is still writing to the pipe, close our read end in
   * attempt to stop it.  */
  CloseHandle (child_stdout_rd);

  DWORD exit_code = EXIT_FAILURE;
  DWORD wait_rc = WaitForSingleObject (process_info.hProcess, INFINITE);
  if (wait_rc != WAIT_FAILED)
    if (!GetExitCodeProcess (process_info.hProcess, &exit_code))
      exit_code = EXIT_FAILURE;

  CloseHandle (process_info.hProcess);
  CloseHandle (process_info.hThread);
  return exit_code;
#else
#error pocl_run_command_capture_output() is not supported on this platform!
#endif
}
