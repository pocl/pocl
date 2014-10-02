/* pocl_cl.h - local runtime library declarations.

   Copyright (c) 2014 Ville Korhonen
   
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

#include "pocl_mem_management.h"
#include "pocl.h"
#include "utlist.h"

typedef struct _mem_manager
{
  pocl_lock_t event_lock;
  pocl_lock_t cmd_lock;
  cl_event event_list;
  _cl_command_node *volatile cmd_list;
} pocl_mem_manager;


static pocl_mem_manager *mm = NULL;

void pocl_init_mem_manager (void)
{
  static unsigned int init_done = 0;
  static pocl_lock_t pocl_init_lock = POCL_LOCK_INITIALIZER;

  if(!init_done)
    {
      POCL_INIT_LOCK(pocl_init_lock);
      init_done = 1;
    }
  POCL_LOCK(pocl_init_lock);
  if (!mm)
    {
      mm = calloc (1, sizeof (pocl_mem_manager));
      POCL_INIT_LOCK (mm->event_lock);
      POCL_INIT_LOCK (mm->cmd_lock);
    }
  POCL_UNLOCK(pocl_init_lock);
}

cl_event pocl_mem_manager_new_event ()
{
  cl_event ev = NULL;
  POCL_LOCK (mm->event_lock);
  if (ev = mm->event_list)
    {
      LL_DELETE (mm->event_list, ev);
      POCL_UNLOCK (mm->event_lock);
      POCL_INIT_LOCK (ev->pocl_lock);
      ev->pocl_refcount = 1; /* no need to lock because event is not in use */
      return ev;
    }
  POCL_UNLOCK (mm->event_lock);
    
  ev = calloc (1, sizeof (struct _cl_event));
  POCL_INIT_OBJECT(ev);
  ev->pocl_refcount = 1;
  return ev;
}

void pocl_mem_manager_free_event (cl_event event)
{
  POCL_LOCK (mm->event_lock);
  LL_PREPEND (mm->event_list, event);
  POCL_UNLOCK(mm->event_lock);
}

_cl_command_node* pocl_mem_manager_new_command ()
{
  _cl_command_node *cmd = NULL;
  POCL_LOCK (mm->cmd_lock);
  if (cmd = mm->cmd_list)
    LL_DELETE (mm->cmd_list, cmd);
  POCL_UNLOCK (mm->cmd_lock);
  
  if (cmd)
    return cmd;
  
  return calloc (1, sizeof (_cl_command_node));
}

void pocl_mem_manager_free_command ( _cl_command_node *cmd_ptr)
{
  POCL_LOCK (mm->cmd_lock);
  LL_PREPEND (mm->cmd_list, cmd_ptr);
  POCL_UNLOCK(mm->cmd_lock);
}
