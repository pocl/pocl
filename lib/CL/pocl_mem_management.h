/* pocl_mem_management.c - manage allocation of runtime objects

   Copyright (c) 2014 Ville Korhonen

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

#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef USE_POCL_MEMMANAGER

void pocl_init_mem_manager (void);

cl_event pocl_mem_manager_new_event (void);

void pocl_mem_manager_free_event (cl_event event);

_cl_command_node* pocl_mem_manager_new_command (void);

void pocl_mem_manager_free_command (_cl_command_node *cmd_ptr);

event_node* pocl_mem_manager_new_event_node ();

void pocl_mem_manager_free_event_node (event_node *ed);

#else

#define pocl_init_mem_manager() NULL

cl_event pocl_mem_manager_new_event ();

#define pocl_mem_manager_free_event(event) POCL_MEM_FREE(event)

#define pocl_mem_manager_new_command() \
  (_cl_command_node*) calloc (1, sizeof (_cl_command_node))

#define pocl_mem_manager_free_command(cmd) POCL_MEM_FREE(cmd)

#define pocl_mem_manager_new_event_node() \
  (event_node*) calloc (1, sizeof (event_node))

#define pocl_mem_manager_free_event_node(en) POCL_MEM_FREE(en)


#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
