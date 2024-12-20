/*
   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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

/* A regression test (#1735) for a self referential macro over an OpenCL
 * built-in:  */

/*   error: (...)/tempfile_TnTaja.cl:2:40 */
/*      <Spelling=(...)/tempfile_TnTaja.cl:1:30>: use of undeclared */
/*      identifier 'sub_group_reduce_add' */
/*   warning: (...)/tempfile_TnTaja.cl:1:9: 'sub_group_reduce_add' macro */
/*      redefined */
/*   warning: (...)/tempfile_TnTaja.cl:1:9: 'sub_group_reduce_add' macro */
/*      redefined */

#define sub_group_reduce_add sub_group_reduce_add

kernel void
k (global int *d)
{
  unsigned id = get_global_id (0);
  d[id] = sub_group_reduce_add (d[id]);
}
