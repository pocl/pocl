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
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The number of available devices. */
extern unsigned int pocl_num_devices;

/**
 * Populates the pocl_devices with the wanted device types.
 *
 * Should be called before accessing the device list. Can be called repeatedly.
 * The devices are shared across contexts, thus must implement resource
 * management internally also across multiple contexts.
 */
cl_int pocl_init_devices();

cl_int pocl_uninit_devices ();

/**
 * \brief Get the count of devices for a specific type
 * \param device_type the device type for which we want the count of devices
 * \return the count of devices for this type
 */
unsigned int pocl_get_device_type_count(cl_device_type device_type);

/**
 * \brief Get a certain amount of devices for a specific type
 * \param type Type of devices wanted
 * \param devices Array of pointer to devices
 * \param num_devices Number of devices queried
 * \return The real number of devices added to devices array which match the specified type
 */
unsigned int pocl_get_devices(cl_device_type device_type, struct _cl_device_id **devices, unsigned int num_devices);

/**
 * \brief Return the count of a specific device in the env var POCL_DEVICES
 * \param dev_type a string describing the device ("basic" for instance)
 * \return If the env var was not set, return -1, if the env var is specified, return 0
 * or the number of occurrence of dev_type in the env var
 */
POCL_EXPORT
int pocl_device_get_env_count(const char *dev_type);

/**
 * \brief Unique global memory id for devices with distinct memory from the system memory
 * \return Unique global mem id, id > 0. Zero is reserved for shared system memory
 */
int pocl_get_unique_global_mem_id();

/* the environment variable that lists the enabled devices */
#define POCL_DEVICES_ENV "POCL_DEVICES"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCL_DEVICES_H */
