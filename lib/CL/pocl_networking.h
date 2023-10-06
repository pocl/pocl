/* pocl_networking.h - Shared helper functions for working with networks &
   sockets

   Copyright (c) 2023 Jan Solanti / Tampere University

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

#include <stdint.h>

#include "pocl_export.h"

#ifndef POCL_NETWORKING_H
#define POCL_NETWORKING_H

#ifdef __cplusplus
extern "C"
{
#endif

  /*
   * Helper for calling getaddrinfo. Returns NULL on error, with the error code
   * in 'error'. Human-readable explanations can be obtained by passing the
   * error code to gai_strerror(). The caller is responsible for freeing the
   * addrinfo chain by calling freeaddrinfo() on the first item of the chain.
   */
  extern struct addrinfo *pocl_resolve_address (const char *address,
                                                uint16_t port, int *error);

  /*
   * Set socket options for PoCL-Remote client connections
   * \param fd the socket fd to set options on
   * \param bufsize size of the desired driver-side send and receive buffers
   * \param is_fast whether this is the "fast" or "slow" socket of the
   * connection
   */
  extern int pocl_remote_client_set_socket_options (int fd, int bufsize,
                                                    int is_fast);

#ifdef __cplusplus
}
#endif

#endif /* POCL_NETWORKING_H */
