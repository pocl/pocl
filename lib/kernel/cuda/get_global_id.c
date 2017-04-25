/* OpenCL built-in library: get_global_id() for CUDA

   Copyright (c) 2016 James Price

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

uint get_nvvm_ntid_x();
uint get_nvvm_ntid_y();
uint get_nvvm_ntid_z();

uint get_nvvm_ctaid_x();
uint get_nvvm_ctaid_y();
uint get_nvvm_ctaid_z();

uint get_nvvm_tid_x();
uint get_nvvm_tid_y();
uint get_nvvm_tid_z();

extern uint _global_offset_x;
extern uint _global_offset_y;
extern uint _global_offset_z;

size_t _CL_OVERLOADABLE
get_global_id(unsigned int dimindx)
{
  switch(dimindx)
    {
    case 0: return get_nvvm_ntid_x() * get_nvvm_ctaid_x() + get_nvvm_tid_x() + _global_offset_x;
    case 1: return get_nvvm_ntid_y() * get_nvvm_ctaid_y() + get_nvvm_tid_y() + _global_offset_y;
    case 2: return get_nvvm_ntid_z() * get_nvvm_ctaid_z() + get_nvvm_tid_z() + _global_offset_z;
    default: return 0;
    }
}
