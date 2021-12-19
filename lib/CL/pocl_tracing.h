/* pocl_tracing.h: interface for event update and tracing system

   Copyright (c) 2015 Clément Léger / Kalray

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

#ifndef POCL_TRACING_H
#define POCL_TRACING_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

int pocl_is_tracing_enabled ();

void pocl_event_updated (cl_event event, int new_status);

/* Initializes the event tracing system selected with POCL_TRACING. */
void pocl_event_tracing_init ();
/* Stops event tracing system */
void pocl_event_tracing_finish ();

/* Struct of trace handlers. */
struct pocl_event_tracer
{
  /* Tracer name used to match POCL_TRACING=xxx env var */
  const char *name;
  /* Init function called when the tracer is matched */
  void (*init) ();
  /* Destroy function called when the tracer is matched */
  void (*destroy) ();
  /* Callback called when an event has been updated */
  void (*event_updated) (cl_event /* event */ , int /* status */ );
};

#ifdef __cplusplus
}
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif				/* POCL_TRACING_H */
