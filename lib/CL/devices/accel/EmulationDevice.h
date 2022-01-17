/* EmulationDevice.hh - basic way of accessing accelerator memory.
 *                 as a memory mapped region

   Copyright (c) 2019-2021 Pekka Jääskeläinen / Tampere University

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

#ifndef EMULATIONDEVICE_H
#define EMULATIONDEVICE_H

#include <pthread.h>

#include "Device.h"

#define EMULATING_ADDRESS 0xE
#define EMULATING_MAX_SIZE (256 * 1024 * 1024)
//#define EMULATING_MAX_SIZE 4 * 4096

struct emulation_data_t
{
  void *emulating_address;
  volatile int emulate_exit_called;
  volatile int emulate_init_done;
};

#ifdef __cplusplus
extern "C"
{
#endif

  void *emulate_accel (void *E_void);

#ifdef __cplusplus
}
#endif


class EmulationDevice : public Device
{
public:
  EmulationDevice ();
  ~EmulationDevice ();
private:
  struct emulation_data_t E;
  pthread_t emulate_thread;
};

#endif
