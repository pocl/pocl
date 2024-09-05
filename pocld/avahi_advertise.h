/* avahi_advertise.h - part of pocl-daemon that performs mDNS on local network
 to advertise the remote server and its devices.


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

#ifndef AVAHI_ADVERTISE_H
#define AVAHI_ADVERTISE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <CL/cl.h>
  void init_avahi_advertisement (const char *server_name,
                                 size_t name_len,
                                 cl_int if_index,
                                 cl_int ip_proto,
                                 uint16_t port,
                                 const char *info,
                                 size_t info_len);

#ifdef __cplusplus
}
#endif

#endif