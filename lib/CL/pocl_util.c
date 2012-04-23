/* OpenCL runtime library: pocl_util utility functions

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include <stdlib.h>
#include <string.h>

#include "pocl_util.h"
#include "pocl_cl.h"
#include "utlist.h"

#define TEMP_DIR_PATH_CHARS 16

void remove_directory (const char *path_name) 
{
  int str_size = 8 + strlen(path_name) + 1;
  char *cmd = (char*)malloc(str_size);
  snprintf (cmd, str_size, "rm -fr %s", path_name);
  system (cmd);
  free (cmd);
}

char *pocl_create_temp_dir() 
{  
  struct temp_dir *td; 
  char *path_name = (struct temp_dir*)malloc (TEMP_DIR_PATH_CHARS);
  assert (path_name != NULL);
  strncpy (path_name, "/tmp/poclXXXXXX\0", TEMP_DIR_PATH_CHARS);
  mkdtemp (path_name);  
  return path_name;
}
