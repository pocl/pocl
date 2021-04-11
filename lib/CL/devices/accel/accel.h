/* accel.h - generic/example driver for hardware accelerators with memory
   mapped control.

   Copyright (c) 2019 Pekka Jääskeläinen / Tampere University

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

#ifndef POCL_ACCEL_H
#define POCL_ACCEL_H

#include "pocl_cl.h"
#include "prototypes.inc"

#ifdef __cplusplus
extern "C"
{
#endif

  GEN_PROTOTYPES (accel)

#ifdef __cplusplus
}
#endif

#define EMULATING_ADDRESS 0xE
#define EMULATING_MAX_SIZE 4 * 4096

void *emulate_accel (void *base_address);

int Emulating = 0;
pthread_t emulate_thread;
volatile int emulate_exit_called = 0;
volatile int emulate_init_done = 0;

#endif /* POCL_ACCEL_H */
