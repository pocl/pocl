/* network_discovery.h - part of pocl-remote driver that performs network
   discovery to find remote servers and their devices.


   Copyright (c) 2023-2024 Yashvardhan Agarwal / Tampere University

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

#ifndef NETWORK_DISCOVERY_H
#define NETWORK_DISCOVERY_H

#include "pocl_networking.h"
#include <CL/cl.h>

#define POCL_REMOTE_DNS_SRV_TYPE_ENV "_pocl._tcp"
#define POCL_REMOTE_SEARCH_DOMAINS_ENV "POCL_REMOTE_SEARCH_DOMAINS"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#define SERVER_ID_SIZE 32

cl_int init_network_discovery (cl_int (*add_discovered_device) (const char *,
                                                                unsigned),
                               cl_int (*reconnect_callback) (const char *),
                               unsigned pocl_dev_type_idx);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif