/* Command queue management functions

   Copyright (c) 2015 Giuseppe Bilotta
   
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

/* We keep a global list of all 'live' command queues in order to be able
 * to force a clFinish on all of them before this is triggered by the destructors
 * at program end, which happen in unspecified order and might cause all sorts
 * of issues. This header defines the signatures of the available functions
 */

#ifndef POCL_QUEUE_H
#define POCL_QUEUE_H

#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef __cplusplus
extern "C" {
#endif

void pocl_init_queue_list();
void pocl_queue_list_insert(cl_command_queue );
void pocl_queue_list_delete(cl_command_queue );

#ifdef __cplusplus
}
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#endif
