/* OpenCL built-in library: length()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

float _CL_OVERLOADABLE length(float x)
{
  return x;
}

float _CL_OVERLOADABLE length(float2 v)
{
  return hypot(v.x, v.y);
}

float _CL_OVERLOADABLE length(float3 v)
{
  return hypot(hypot(v.x, v.y), v.z);
}

float _CL_OVERLOADABLE length(float4 v)
{
  return hypot(hypot(hypot(v.x, v.y), v.z), v.w);
}

double _CL_OVERLOADABLE length(double x)
{
  return x;
}

double _CL_OVERLOADABLE length(double2 v)
{
  return hypot(v.x, v.y);
}

double _CL_OVERLOADABLE length(double3 v)
{
  return hypot(hypot(v.x, v.y), v.z);
}

double _CL_OVERLOADABLE length(double4 v)
{
  return hypot(hypot(hypot(v.x, v.y), v.z), v.w);
}
