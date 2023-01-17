/* basic.h - a minimalistic single core pocl device driver layer implementation

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2020 Pekka Jääskeläinen / Tampere University

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
/**
 * @file basic.h
 *
 * The purpose of the 'basic' device driver is to serve as an example of
 * a minimalistic (but still working) device driver for pocl.
 *
 * It is a "native device" without multithreading and uses the malloc
 * directly for buffer allocation etc. It also executes work groups in
 * their sequential (increasing) order, thus makes it useful as a test
 * device.
 */

#ifndef POCL_BASIC_H
#define POCL_BASIC_H

#include "pocl_cl.h"

#include "prototypes.inc"
GEN_PROTOTYPES (basic)

cl_int pocl_basic_device_info_ext (cl_device_id device,
                                   cl_device_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret);

#endif /* POCL_BASIC_H */
