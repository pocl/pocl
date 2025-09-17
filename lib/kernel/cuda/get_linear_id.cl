/* OpenCL built-in library: get_{local,global}_linear_id()

   Copyright (c) 2025 Aritra Bhakat / I-Conic Vision

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

uint get_nvvm_nctaid_x();
uint get_nvvm_nctaid_y();
uint get_nvvm_nctaid_z();

uint get_nvvm_ntid_x();
uint get_nvvm_ntid_y();
uint get_nvvm_ntid_z();

uint get_nvvm_ctaid_x();
uint get_nvvm_ctaid_y();
uint get_nvvm_ctaid_z();

uint get_nvvm_tid_x();
uint get_nvvm_tid_y();
uint get_nvvm_tid_z();


size_t _CL_OVERLOADABLE _CL_READNONE
get_global_linear_id ()
{
  return ((get_nvvm_ntid_z() * get_nvvm_ctaid_z() + get_nvvm_tid_z())
          * get_nvvm_ntid_y() * get_nvvm_nctaid_y() * get_nvvm_ntid_x() * get_nvvm_nctaid_x())

         + ((get_nvvm_ntid_y() * get_nvvm_ctaid_y() + get_nvvm_tid_y())
            * get_nvvm_ntid_x() * get_nvvm_nctaid_x())

         + (get_nvvm_ntid_x() * get_nvvm_ctaid_x() + get_nvvm_tid_x());
}

size_t _CL_OVERLOADABLE _CL_READNONE
get_local_linear_id (void)
{
  return (get_nvvm_tid_z() * get_nvvm_ntid_y() * get_nvvm_ntid_x())
         + (get_nvvm_tid_y() * get_nvvm_ntid_x()) + get_nvvm_tid_x();
}
