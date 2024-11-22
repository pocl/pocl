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

#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <CL/opencl.hpp>

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

    {
        std::vector<cl::Platform> PlatformList;
        cl::Platform::get(&PlatformList);
        POCL_MSG_PRINT_INFO("showing devices on: %" PRIuS " platform(s) \n", PlatformList.size());
        for(size_t i = 0; i< PlatformList.size(); i++) {

            std::string platformName = PlatformList[i].getInfo<CL_PLATFORM_NAME>();
            POCL_MSG_PRINT_INFO("%" PRIuS": %s\n", i, platformName.c_str());

            std::vector<cl::Device> DeviceList;
            PlatformList[i].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
            for(size_t j = 0; j<DeviceList.size(); j++) {
                std::string deviceName = DeviceList[j].getInfo<CL_DEVICE_NAME>();
                POCL_MSG_PRINT_INFO("\t%" PRIuS": %s \n", j, deviceName.c_str());
            }

            // also query all the custom devices that don't show up under
            // device type all
            size_t allSize = DeviceList.size();
            PlatformList[i].getDevices(CL_DEVICE_TYPE_CUSTOM, &DeviceList);
            for(size_t j = 0; j<DeviceList.size(); j++) {
                std::string deviceName = DeviceList[j].getInfo<CL_DEVICE_NAME>();
                POCL_MSG_PRINT_INFO("\t%" PRIuS": %s \n", allSize+j, deviceName.c_str());
            }
        }
    }

  int error;
  PoclDaemon server;
  bool UseVsock = false;
  if (ai.vsock_flag == 1) {
#ifdef HAVE_LINUX_VSOCK_H
    UseVsock = true;
#else
    POCL_MSG_ERR("This pocld was built without vsock support\n");
    return -1;
#endif
  } else {
    UseVsock = false;
  }
  if ((error = server.launch(
           std::string(ai.address_arg ? ai.address_arg : ""),
           listen_ports, UseVsock)))
    return error;

  server.waitForExit();
  return 0;
}
