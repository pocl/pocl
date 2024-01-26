/* pocld.cc - starting point and "main" loop of pocld

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2024 Jan Solanti / Tampere University

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

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netdb.h>

#ifdef __linux__
#include <dlfcn.h>
#include <ifaddrs.h>
#include <libgen.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/time.h>
#endif

#include "pocl_debug.h"
#include "pocl_networking.h"
#include "pocl_remote.h"
#include "pocld_config.h"

#include "cmdline.h"
#include "daemon.hh"

in_addr_t find_default_ip_address() {
  struct in_addr listen_addr;
  listen_addr.s_addr = inet_addr("127.0.0.1");

#ifdef __linux__
  struct ifaddrs *ifa = NULL;
  int err = getifaddrs(&ifa);

  if (err == 0 && ifa) {
    struct ifaddrs *p;

    for (p = ifa; p != NULL; p = p->ifa_next) {
      if ((p->ifa_flags & IFF_UP) == 0)
        continue;
      if (p->ifa_flags & IFF_LOOPBACK)
        continue;
      if ((p->ifa_flags & IFF_RUNNING) == 0)
        continue;

      struct sockaddr *saddr = p->ifa_addr;
      if (saddr->sa_family != AF_INET)
        continue;

      struct sockaddr_in *saddr_in = (struct sockaddr_in *)saddr;
      if (saddr_in->sin_addr.s_addr == 0)
        continue;
      else {
        listen_addr.s_addr = saddr_in->sin_addr.s_addr;
        break;
      }
    }
    freeifaddrs(ifa);
  } else
    POCL_MSG_ERR("getifaddrs() failed or returned no data.\n");
#endif

  return listen_addr.s_addr;
}

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
  /* NOTE: struct sockaddr does NOT have enough space for all address types */
  struct sockaddr_storage base_addr = {};
  socklen_t base_addrlen;
  if (ai.address_arg) {
    addrinfo *resolved_address =
        pocl_resolve_address(ai.address_arg, listen_ports.command, &error);
    if (!error) {
      /* Unfortunately getaddrinfo does not guarantee that the returned
       * addresses would actually WORK. For example misconfigured IPv4-only
       * setups sometimes return IPv6 addresses that are not bindable. As a
       * workaround, iterate over the returned addresses until one is found that
       * works. */
      bool found_bindable_address = false;
      std::vector<int> bind_errors;
      for (addrinfo *ai = resolved_address;
           ai != nullptr && !found_bindable_address; ai = ai->ai_next) {
        memcpy(&base_addr, resolved_address->ai_addr,
               resolved_address->ai_addrlen);
        base_addrlen = resolved_address->ai_addrlen;
        int tmp = socket(ai->ai_family, SOCK_STREAM, IPPROTO_TCP);
        if (!tmp)
          continue;
        error = bind(tmp, ai->ai_addr, ai->ai_addrlen);
        if (!error)
          found_bindable_address = true;
        else
          bind_errors.push_back(errno);
        close(tmp);
      }
      freeaddrinfo(resolved_address);
      if (!found_bindable_address) {
        POCL_MSG_ERR("Requested listen address %s did not resolve to any "
                     "bindable address:\n",
                     ai.address_arg);
        int idx = 0;
        for (int e : bind_errors)
          POCL_MSG_ERR("resolved address %d: %s\n", idx++, strerror(e));
        return EXIT_FAILURE;
      }
    } else {
      POCL_MSG_ERR("Failed to resolve listen address: %s\n",
                   gai_strerror(error));
      return EXIT_FAILURE;
    }
  } else {
    struct sockaddr_in *fallback = (struct sockaddr_in *)&base_addr;
    fallback->sin_family = AF_INET;
    fallback->sin_addr.s_addr = find_default_ip_address();
    base_addrlen = sizeof(struct sockaddr_in);
  }

  PoclDaemon server;
  if ((error = server.launch(base_addr, base_addrlen, listen_ports)))
    return error;

  server.waitForExit();
  return 0;
}
