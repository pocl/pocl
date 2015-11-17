/* OpenCL built-in library: fast_length()

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

/*
 * TODO these should use "half_sqrt" but i'm not sure if 1) it exists in HSAIL,
 * and 2) if it'd be any faster than native sqrt()
 */

float _CL_OVERLOADABLE fast_length(float x)
{
  return x;
}

float _CL_OVERLOADABLE fast_length(float2 v)
{
  float temp = v.x * v.x;
  temp = fma(v.y, v.y, temp);
  return sqrt(temp);
}

float _CL_OVERLOADABLE fast_length(float3 v)
{
  float temp = v.x * v.x;
  temp = fma(v.y, v.y, temp);
  temp = fma(v.z, v.z, temp);
  return sqrt(temp);
}

float _CL_OVERLOADABLE fast_length(float4 v)
{
  float temp = v.x * v.x;
  temp = fma(v.y, v.y, temp);
  temp = fma(v.z, v.z, temp);
  temp = fma(v.w, v.w, temp);
  return sqrt(temp);
}

double _CL_OVERLOADABLE fast_length(double x)
{
  return x;
}

double _CL_OVERLOADABLE fast_length(double2 v)
{
  double temp = v.x * v.x;
  temp = fma(v.y, v.y, temp);
  return sqrt(temp);
}

double _CL_OVERLOADABLE fast_length(double3 v)
{
  double temp = v.x * v.x;
  temp = fma(v.y, v.y, temp);
  temp = fma(v.z, v.z, temp);
  return sqrt(temp);
}

double _CL_OVERLOADABLE fast_length(double4 v)
{
  double temp = v.x * v.x;
  temp = fma(v.y, v.y, temp);
  temp = fma(v.z, v.z, temp);
  temp = fma(v.w, v.w, temp);
  return sqrt(temp);
}
