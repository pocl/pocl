/* pocld.cc - starting point and "main" loop of pocld

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University

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

#include <csignal>
#include <cstdlib>
#include <unistd.h>

#ifdef __linux__
#include <sys/resource.h>
#endif

#include "pocl_debug.h"
#include "pocld_config.h"

#include "cmdline.h"
#include "daemon.hh"

int main(int argc, char *argv[]) {
  struct gengetopt_args_info ai;
  memset(&ai, 0, sizeof(struct gengetopt_args_info));

  if (cmdline_parser(argc, argv, &ai) != 0) {
    exit(1);
  }

#ifdef POCL_DEBUG_MESSAGES
  const char *logfilter = NULL;
  if (ai.log_filter_arg)
    logfilter = ai.log_filter_arg;
  else
    logfilter = getenv("POCLD_LOGLEVEL");
  if (!logfilter)
    logfilter = "";
  pocl_stderr_is_a_tty = isatty(fileno(stderr));
  pocl_debug_messages_setup(logfilter);
#endif

#ifdef __linux__
  // to avoid cores at abort
  struct rlimit core_limit;
  core_limit.rlim_cur = 0;
  core_limit.rlim_max = 0;
  if (setrlimit(RLIMIT_CORE, &core_limit) != 0)
    POCL_MSG_ERR("setting rlimit_core failed!\n");
#endif

  // ignore sigpipe, because we don't want to shutdown on closed sockets
  signal(SIGPIPE, SIG_IGN);

  struct ServerPorts listen_ports = {};
  if (ai.port_arg)
    listen_ports.command = (unsigned short)ai.port_arg;
  else
    listen_ports.command = DEFAULT_POCL_REMOTE_PORT;

  assert(listen_ports.command > 0);
  listen_ports.stream = listen_ports.command + 1;
  listen_ports.peer = listen_ports.command + 2;
#ifdef ENABLE_RDMA
  listen_ports.peer_rdma = listen_ports.command + 3;
  listen_ports.rdma = listen_ports.command + 4;
#endif

  int error;
  PoclDaemon server;
  if ((error = server.launch(
           std::move(std::string(ai.address_arg ? ai.address_arg : "")),
           listen_ports)))
    return error;

  server.waitForExit();
  return 0;
}
