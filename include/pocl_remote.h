/* pocl_remote.h - configuration defines for the remote backend

   Copyright (c) 2018 Michal Babej / Tampere University of Technology

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

#ifndef POCL_REMOTE_SHARED_H
#define POCL_REMOTE_SHARED_H

#if defined(ENABLE_REMOTE_DISCOVERY_ANDROID)
#include <CL/cl_platform.h>
#endif

#define DEFAULT_POCL_REMOTE_PORT 10998

#define MAX_REMOTE_DEVICES 512

#define MAX_REMOTE_PARAM_LENGTH 256

#if defined(ENABLE_REMOTE_DISCOVERY_ANDROID)
#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Helper function used to dynamically add a remote server and its devices.
   * It is currently only used by android where the network discovery is
   * performed using android network libraries and the discovered servers are
   * added using this function. The function is implemented in the remote
   * driver in network_discovery.c.
   *
   * \param id [in] Unique ID with which server advertises itself.
   * \param domain [in] Domain name in which the server was found.
   * \param server_key [in] Combination of "IP:port"
   * \param type [in] Type of the discovered service. Eg type: "_pocl._tcp"
   * \param device_count [in] Number of devices in the remote server.
   */
  void pocl_remote_discovery_add_server (const char *id,
                                         const char *domain,
                                         const char *server_key,
                                         const char *type,
                                         cl_uint device_count);

#ifdef __cplusplus
}
#endif
#endif

#endif
