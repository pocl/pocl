/* Correctly-rounded cubic root of binary16 value.

Copyright (c) 2025 Maxence Ponsardin.

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "common_types.h"

static constant ushort tm[] =
  {0x00, 0xff, 0xff, 0xff,
	 0x33, 0x1c, 0x32, 0xff,
	 0xff, 0xff, 0xff, 0x00,
	 0xff, 0xff, 0xff, 0x10};
static constant int te[] =
  {0, 0, 0, 0,
	 1, 2, 0, 0,
	 0, 0, 0, 1,
	 0, 0, 0, 0};
static constant ushort tf[] =
  {0x3c00, 0, 0, 0,
	 0x3d80, 0x3f00, 0x3c80, 0,
	 0, 0, 0, 0x3e00,
	 0, 0, 0, 0x3d00};

_CL_OVERLOADABLE half cbrt(half x){

	b16u16_u t = {.f = x};
	b32u32_u xf = {.f = x};
	if (tm[(xf.u >> 19) % 16] == (xf.u >> 13) % 64) { // exact cases (not supported by cbrtf)
		int expo = (xf.u & 0x7fffffff) >> 23;
		if (te[(xf.u >> 19) % 16] == (expo + 2) % 3) {
			t.u = (t.u & 0x8000) + ((unsigned)((expo - 127 - te[(xf.u >> 19) % 16]) / 3) << 10) + tf[(xf.u >> 19) % 16];
			return t.f;
		}
	}
	if ((t.u & 0x03ff) == 0x0151) { // only wrong case is 0x1.544pk with k = 1 mod 3
		int expo = (t.u & 0x7fff) >> 10;
		if (expo % 3 == 1 && expo < 31) { // avoid sNaN and k != 1 mod 3
			t.u = (((expo - 16) / 3 + 15) << 10) + 0x018b + ((t.u >> 15) << 15);
			return (float) t.f + 0x1p-16f * ((t.u >> 15) - 0.5f);
		}
	}
	return (half) cbrt ((float) x);
}
