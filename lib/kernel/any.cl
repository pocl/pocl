/* OpenCL built-in library: any()

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
   FITNESS FOR A PARTICULAR PURPOSE AND NONORDEREDRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

int _cl_overloadable any(char a)
{
  return a < (char)0;
}

int _cl_overloadable any(char2 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(char3 a)
{
  return any(a.s01) || any(a.s2);
}

int _cl_overloadable any(char4 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(char8 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(char16 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(short a)
{
  return a < (short)0;
}

int _cl_overloadable any(short2 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(short3 a)
{
  return any(a.s01) || any(a.s2);
}

int _cl_overloadable any(short4 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(short8 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(short16 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(int a)
{
  return a < 0;
}

int _cl_overloadable any(int2 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(int3 a)
{
  return any(a.s01) || any(a.s2);
}

int _cl_overloadable any(int4 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(int8 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(int16 a)
{
  return any(a.lo) || any(a.hi);
}

#ifdef cles_khr_int64
int _cl_overloadable any(long a)
{
  return a < 0L;
}

int _cl_overloadable any(long2 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(long3 a)
{
  return any(a.s01) || any(a.s2);
}

int _cl_overloadable any(long4 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(long8 a)
{
  return any(a.lo) || any(a.hi);
}

int _cl_overloadable any(long16 a)
{
  return any(a.lo) || any(a.hi);
}
#endif
