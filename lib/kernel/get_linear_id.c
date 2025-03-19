/* OpenCL built-in library: get_{local,global}_linear_id()

   Copyright (c) 2022 Michal Babej / Tampere University

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

extern const size_t _local_size_x;
extern const size_t _local_size_y;
extern const size_t _local_size_z;

extern const size_t _group_id_x;
extern const size_t _group_id_y;
extern const size_t _group_id_z;

extern const size_t _local_id_x;
extern const size_t _local_id_y;
extern const size_t _local_id_z;

extern const size_t _global_offset_x;
extern const size_t _global_offset_y;
extern const size_t _global_offset_z;

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_local_id (unsigned int dimindx);

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_global_size (unsigned int dimindx);

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_local_size (unsigned int dimindx);

/* attribute optnone disables all optimizations.
 * This was necessary, because running opt on kernel library
 * introduced global "switch tables" (@switch.table.XX)
 * which referenced the global variables like @_global_offset*,
 * and this was preventing these global vars from being optimized
 * out after privatizeContext() in Workgroup pass. Leading to
 * undefined references in final .so
 */

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE get_global_linear_id (void);
size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE get_local_linear_id (void);

#if _MSC_VER
size_t _CL_READNONE _CL_OPTNONE
__identifier ("?get_global_linear_id@@$$J0YAKXZ") ()
#else
size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_global_linear_id ()
#endif
{
  return ((_local_size_z * _group_id_z + get_local_id (2))
          * get_global_size (1) * get_global_size (0))

         + ((_local_size_y * _group_id_y + get_local_id (1))
            * get_global_size (0))

         + (_local_size_x * _group_id_x + get_local_id (0));
}

#if _MSC_VER
size_t _CL_READNONE _CL_OPTNONE
__identifier ("?get_local_linear_id@@$$J0YAKXZ") ()
#else
size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_local_linear_id (void)
#endif
{
  return (get_local_id (2) * get_local_size (1) * get_local_size (0))
         + (get_local_id (1) * get_local_size (0)) + get_local_id (0);
}
