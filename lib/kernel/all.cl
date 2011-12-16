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

int _cl_overloadable all(char a)
{
  return a < (char)0;
}

int _cl_overloadable all(char2 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(char3 a)
{
  return all(a.s01) && all(a.s2);
}

int _cl_overloadable all(char4 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(char8 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(char16 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(short a)
{
  return a < (short)0;
}

int _cl_overloadable all(short2 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(short3 a)
{
  return all(a.s01) && all(a.s2);
}

int _cl_overloadable all(short4 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(short8 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(short16 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(int a)
{
  return a < 0;
}

int _cl_overloadable all(int2 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(int3 a)
{
  return all(a.s01) && all(a.s2);
}

int _cl_overloadable all(int4 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(int8 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(int16 a)
{
  return all(a.lo) && all(a.hi);
}

#ifdef cles_khr_int64
int _cl_overloadable all(long a)
{
  return a < 0L;
}

int _cl_overloadable all(long2 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(long3 a)
{
  return all(a.s01) && all(a.s2);
}

int _cl_overloadable all(long4 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(long8 a)
{
  return all(a.lo) && all(a.hi);
}

int _cl_overloadable all(long16 a)
{
  return all(a.lo) && all(a.hi);
}
#endif
