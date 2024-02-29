/* pocl_networking.c - Shared helper functions for working with networks and
   sockets

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

#include <errno.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>

#include "pocl_debug.h"
#include "pocl_networking.h"

struct addrinfo *
pocl_resolve_address (const char *address, uint16_t port, int *error)
{
  struct addrinfo hint;
  memset (&hint, 0, sizeof (hint));
  hint.ai_family = AF_UNSPEC;
  hint.ai_socktype = SOCK_STREAM;

  int is_numeric = 0;
#ifdef ANDROID
  /* Check whether address is an IP or a host name since Android apparently
   * can't resolve IP addresses without working DNS unless explicitly told
   * that it is a numeric address. This heuristic is of course rather brittle
   * but having a domain name that happens to be fully representable as hex
   * digits is hopefully less common than ipv6 addresses. */
  is_numeric = 1;
  for (const char *c = address; c && *c != 0; ++c)
    {
      if (!isxdigit (*c) && *c != '.' && *c != ':' && *c != '[' && *c != ']')
        {
          is_numeric = 0;
        }
    }
#endif

  hint.ai_flags = is_numeric ? AI_NUMERICHOST
                             : (AI_ADDRCONFIG | AI_CANONNAME | AI_V4MAPPED);
  if (address == NULL)
    hint.ai_flags = AI_PASSIVE;
  hint.ai_flags |= AI_NUMERICSERV;

  struct addrinfo *info = NULL;
  char portstr[6] = {};
  snprintf (portstr, 6, "%5d", port);

  int err = getaddrinfo (address, portstr, &hint, &info);
  if (error)
    *error = err;

  return info;
}

#define SECONDS_TO_MS 1000
int
pocl_remote_client_set_socket_options (int socket_fd, int bufsize, int is_fast,
                                       int ai_family)
{
  const int one = 1;
  unsigned int user_timeout = 10 * SECONDS_TO_MS;
  struct timeval tv;
  tv.tv_sec = user_timeout;
  tv.tv_usec = 0;
  int retries = 5;

#ifdef SO_PRIORITY
  // 1- low priority, 7 - high priority (7 reserved for root)
  int prio = 0; // valid values are in the range [1,7]
  if (is_fast)
    {
      prio = 6;
    }
  else
    {
      prio = 1;
    }
  POCL_RETURN_ERROR_ON (
      (setsockopt (socket_fd, SOL_SOCKET, SO_PRIORITY, &prio, sizeof (prio))),
      -1, "setsockopt(SO_PRIORITY) returned errno: %i\n", errno);
#endif

  POCL_RETURN_ERROR_ON ((setsockopt (socket_fd, SOL_SOCKET, SO_RCVBUF,
                                     &bufsize, sizeof (bufsize))),
                        -1, "setsockopt(SO_RCVBUF) returned errno: %i\n",
                        errno);

  POCL_RETURN_ERROR_ON ((setsockopt (socket_fd, SOL_SOCKET, SO_SNDBUF,
                                     &bufsize, sizeof (bufsize))),
                        -1, "setsockopt(SO_SNDBUF) returned errno: %i\n",
                        errno);

  POCL_RETURN_ERROR_ON (
      (setsockopt (socket_fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof (one))),
      -1, "setsockopt(SO_KEEPALIVE) returned errno: %i\n", errno);

  POCL_RETURN_ERROR_ON ((setsockopt (socket_fd, SOL_SOCKET, SO_RCVTIMEO,
                                     (const char *)&tv, sizeof (tv))),
                        -1, "setsockopt(SO_RCVTIMEO) returned errno: %i\n",
                        errno);
  POCL_RETURN_ERROR_ON ((setsockopt (socket_fd, SOL_SOCKET, SO_SNDTIMEO,
                                     (const char *)&tv, sizeof (tv))),
                        -1, "setsockopt(SO_SNDTIMEO) returned errno: %i\n",
                        errno);

#ifdef ENABLE_VSOCK
  if (ai_family == AF_VSOCK)
    {
      return 0;
    }
#endif
  POCL_RETURN_ERROR_ON (
      (setsockopt (socket_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof (one))),
      -1, "setsockopt(TCP_NODELAY) returned errno: %i\n", errno);

  // disable delayed_ack on both
#ifdef TCP_QUICKACK
  POCL_RETURN_ERROR_ON (
      (setsockopt (socket_fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof (one))),
      -1, "setsockopt(TCP_QUICKACK) returned errno: %i\n", errno);
#endif

#if defined(TCP_USER_TIMEOUT)
  POCL_RETURN_ERROR_ON (
      (setsockopt (socket_fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_timeout,
                   sizeof (user_timeout))),
      -1, "setsockopt(TCP_USER_TIMEOUT) returned errno: %i\n", errno);
#elif defined(TCP_CONNECTIONTIMEOUT)
  POCL_RETURN_ERROR_ON (
      (setsockopt (socket_fd, IPPROTO_TCP, TCP_CONNECTIONTIMEOUT,
                   &user_timeout, sizeof (user_timeout))),
      -1, "setsockopt(TCP_CONNECTIONTIMEOUT) returned errno: %i\n", errno);
#endif

#ifdef TCP_SYNCNT
  POCL_RETURN_ERROR_ON (setsockopt (socket_fd, IPPROTO_TCP, TCP_SYNCNT,
                                    &retries, sizeof (retries)),
                        -1, "setsockopt(TCP_SYNCNT) returned errno: %i\n",
                        errno);
#endif
  POCL_RETURN_ERROR_ON (setsockopt (socket_fd, IPPROTO_TCP, TCP_KEEPCNT,
                                    &retries, sizeof (retries)),
                        -1, "setsockopt(TCP_KEEPCNT) returned errno: %i\n",
                        errno);
  POCL_RETURN_ERROR_ON (
      setsockopt (socket_fd, IPPROTO_TCP, TCP_KEEPINTVL, &one, sizeof (one)),
      -1, "setsockopt(TCP_KEEPINTVL) returned errno: %i\n", errno);
#ifdef TCP_KEEPIDLE
  POCL_RETURN_ERROR_ON (
      setsockopt (socket_fd, IPPROTO_TCP, TCP_KEEPIDLE, &one, sizeof (one)),
      -1, "setsockopt(TCP_KEEPINTVL) returned errno: %i\n", errno);
#endif

  return 0;
}
