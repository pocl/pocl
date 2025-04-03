#include "pocl_export.h"

#include <stddef.h>

#ifndef POCL_RUN_COMMAND_H
#define POCL_RUN_COMMAND_H

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Wrapper for running commands.
   *
   * \param args The list of arguments terminated by NULL, including the path
   * to the program to be run.
   * \return The return value from the executed command.
   */
  POCL_EXPORT
  int pocl_run_command (const char **args);

  /**
   * Wrapper for running commands with their output captured to a buffer.
   *
   * \todo This currently might block forever if capture_string is too small
   * for all the output.
   *
   * \param capture_string The target for storing the stdout and stderr.
   * \param captured_bytes [in/out] Input the number of bytes allocated in
   *        capture_string, outputs the number of bytes written there.
   * \param args The list of arguments terminated by NULL, including the path
   * to the program to be run.
   * \return The return value from the executed command.
   */
  POCL_EXPORT
  int pocl_run_command_capture_output (char *capture_string,
                                       size_t *captured_bytes,
                                       const char **args);

#ifdef __cplusplus
}
#endif

#endif // POCL_RUN_COMMAND_H
