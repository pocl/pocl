/* OpenCL built-in library: get_global_id()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

extern size_t _local_size_x;
extern size_t _local_size_y;
extern size_t _local_size_z;

extern size_t _group_id_x;
extern size_t _group_id_y;
extern size_t _group_id_z;

extern size_t _local_id_x;
extern size_t _local_id_y;
extern size_t _local_id_z;

extern size_t _global_offset_x;
extern size_t _global_offset_y;
extern size_t _global_offset_z;

size_t _CL_OVERLOADABLE
get_local_id(unsigned int dimindx);

/* attribute optnone disables all optimizations.
 * This was necessary, because running opt on kernel library
 * introduced global "switch tables" (@switch.table.XX)
 * which referenced the global variables like @_global_offset*,
 * and this was preventing these global vars from being optimized
 * out after privatizeContext() in Workgroup pass. Leading to
 * undefined references in final .so
 */

__attribute__ ((optnone, noinline))
size_t _CL_OVERLOADABLE
get_global_id(unsigned int dimindx)
{
  switch(dimindx)
    {
    case 0: return _global_offset_x + _local_size_x * _group_id_x + get_local_id(0);
    case 1: return _global_offset_y + _local_size_y * _group_id_y + get_local_id(1);
    case 2: return _global_offset_z + _local_size_z * _group_id_z + get_local_id(2);
    default: return 0;
    }
}
