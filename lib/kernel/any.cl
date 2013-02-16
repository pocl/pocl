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

int _CL_OVERLOADABLE any(char a)
{
  return a < (char)0;
}

int _CL_OVERLOADABLE any(char2 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(char3 a)
{
  return any(a.s01) || any(a.s2);
}

int _CL_OVERLOADABLE any(char4 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(char8 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(char16 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(short a)
{
  return a < (short)0;
}

int _CL_OVERLOADABLE any(short2 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(short3 a)
{
  return any(a.s01) || any(a.s2);
}

int _CL_OVERLOADABLE any(short4 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(short8 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(short16 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(int a)
{
  return a < 0;
}

int _CL_OVERLOADABLE any(int2 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(int3 a)
{
  return any(a.s01) || any(a.s2);
}

int _CL_OVERLOADABLE any(int4 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(int8 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(int16 a)
{
  return any(a.lo) || any(a.hi);
}

#ifdef cles_khr_int64
int _CL_OVERLOADABLE any(long a)
{
  return a < 0L;
}

int _CL_OVERLOADABLE any(long2 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(long3 a)
{
  return any(a.s01) || any(a.s2);
}

int _CL_OVERLOADABLE any(long4 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(long8 a)
{
  return any(a.lo) || any(a.hi);
}

int _CL_OVERLOADABLE any(long16 a)
{
  return any(a.lo) || any(a.hi);
}
#endif
