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

#ifndef POCL_NETWORKING_H
#define POCL_NETWORKING_H

/** Tag for transport_info_t to indicate the type of the socket */
typedef enum transport_domain_e
{
  /** Unix domain socket */
  TransportDomain_Unix,
  /** IPv4 or IPv6 */
  TransportDomain_Inet,
  /** VSOCK VirtIO socket */
  TransportDomain_Vsock,
} transport_domain_t;

/** Wrapper struct that holds everything the communication (read/write)
 * functions need to function, starting with a tag to indicate what kind of
 * connection this is */
typedef struct transport_info_s
{
  transport_domain_t domain;
  int fd;
} transport_info_t;

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
   * \param ai_family is the ai_family of fd
   */
  extern int pocl_remote_client_set_socket_options (int fd, int bufsize,
                                                    int is_fast,
                                                    int ai_family);
  /**
   * pocl_freeaddrinfo - free addrinfo obtained from host_*() functions
   * @ai: pointer to addrinfo to free
   *
   * The addrinfos returned by pocl_*() functions may not have been allocated
   * by a call to getaddrinfo(3).  It is not safe to free them directly with
   * freeaddrinfo(3).  Use this function instead.
   */
  extern void pocl_freeaddrinfo (struct addrinfo *ai);

  /*
   * vsock_hostname_addrinfo - Custom 'getaddrinfo' for vsock addresses
   *
   * This function resolves a vsock hostname to an addrinfo structure that can
   * be used to create a socket and connect to a vsock service.
   *
   * The hostname should be in the following format:
   *
   *   vsock:[cid]
   *
   * Where:
   *
   *   * `cid` is the context ID of the target virtual machine.
   *
   * This function will allocate memory to store the addrinfo structure and the
   * sockaddr_vm structure. The caller must free the addrinfo structure using
   * the host_freeaddrinfo() function instead of freeaddrinfo(3).
   *
   * Parameters:
   *
   *   * `hostname`: Pointer to a string containing the vsock hostname.
   *   * `port`: The port number of the vsock service.
   *
   * Returns:
   *
   *   * On success, returns a pointer to the addrinfo structure.
   *   * On failure, returns NULL.
   */
  extern struct addrinfo *vsock_hostname_addrinfo (const char *hostname,
                                                   uint16_t port);
#ifdef __cplusplus
}
#endif

#endif /* POCL_NETWORKING_H */
