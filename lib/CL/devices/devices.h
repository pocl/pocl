/* devices.h - OpenCL device type definition.

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2011-2012 Pekka Jääskeläinen / Tampere University of Technology
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef POCL_DEVICES_H
#define POCL_DEVICES_H

#include "../pocl_cl.h"

#pragma GCC visibility push(hidden)

/* The number of available devices. */
extern int pocl_num_devices;
/* The enabled devices. */
extern struct _cl_device_id* pocl_devices;

/**
 * Populates the pocl_devices with the wanted device types.
 *
 * Should be called before accessing the device list. Can be called repeatedly.
 * The devices are shared across contexts, thus must implement resource
 * management internally also across multiple contexts.
 */
void pocl_init_devices();

#pragma GCC visibility pop

/* the environment variable that lists the enabled devices */
#define POCL_DEVICES_ENV "POCL_DEVICES"

#endif /* POCL_DEVICES_H */
