/* OpenCL built-in library: all()

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

int _CL_OVERLOADABLE all(char a)
{
  return a < (char)0;
}

int _CL_OVERLOADABLE all(char2 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(char3 a)
{
  return all(a.s01) && all(a.s2);
}

int _CL_OVERLOADABLE all(char4 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(char8 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(char16 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(short a)
{
  return a < (short)0;
}

int _CL_OVERLOADABLE all(short2 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(short3 a)
{
  return all(a.s01) && all(a.s2);
}

int _CL_OVERLOADABLE all(short4 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(short8 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(short16 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(int a)
{
  return a < 0;
}

int _CL_OVERLOADABLE all(int2 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(int3 a)
{
  return all(a.s01) && all(a.s2);
}

int _CL_OVERLOADABLE all(int4 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(int8 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(int16 a)
{
  return all(a.lo) && all(a.hi);
}

#ifdef cl_khr_int64
int _CL_OVERLOADABLE all(long a)
{
  return a < 0L;
}

int _CL_OVERLOADABLE all(long2 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(long3 a)
{
  return all(a.s01) && all(a.s2);
}

int _CL_OVERLOADABLE all(long4 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(long8 a)
{
  return all(a.lo) && all(a.hi);
}

int _CL_OVERLOADABLE all(long16 a)
{
  return all(a.lo) && all(a.hi);
}
#endif
