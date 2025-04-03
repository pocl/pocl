/* OpenCL built-in library: get_num_groups()

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

extern const size_t _num_groups_x;
extern const size_t _num_groups_y;
extern const size_t _num_groups_z;

#if _MSC_VER
size_t _CL_READNONE _CL_OPTNONE
__identifier ("?get_num_groups@@$$J0YAKI@Z") (unsigned int dimindx)
#else
size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_num_groups (unsigned int dimindx)
#endif
{
  switch(dimindx)
    {
    case 0: return _num_groups_x;
    case 1: return _num_groups_y;
    case 2: return _num_groups_z;
    default:
      return 1;
    }
}

