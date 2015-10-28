/* OpenCL built-in library: lgamma()

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

#include "hsail_templates.h"

#define LN_SQRT2PI (0.91893853320467274178)
#define LN_SQRT2PI_F (0.91893853320467274178f)

double _cl_builtin_lgamma(double x) {
    x -= 1.0;
    if (x <= -1.0)
      return NAN;

    double z = x;
    double a = 0.99999999999999709182;
    z += 1.0;
    a += 57.156235665862923517 / z;
    z +=1.0;
    a += -59.597960355475491248 / z;
    z +=1.0;
    a += 14.136097974741747174 / z;
    z +=1.0;
    a += -0.49191381609762019978 / z;
    z +=1.0;
    a += 0.33994649984811888699e-4 / z;
    z +=1.0;
    a += 0.46523628927048575665e-4 / z;
    z +=1.0;
    a += -0.98374475304879564677e-4 / z;
    z +=1.0;
    a += 0.15808870322491248884e-3 / z;
    z +=1.0;
    a += -0.21026444172410488319e-3 / z;
    z +=1.0;
    a += 0.21743961811521264320e-3 / z;
    z +=1.0;
    a += -0.16431810653676389022e-3 / z;
    z +=1.0;
    a += 0.84418223983852743293e-4 / z;
    z +=1.0;
    a += -0.26190838401581408670e-4 / z;
    z +=1.0;
    a += 0.36899182659531622704e-5 / z;

    double tmp = x + (607/128.0 + 0.5);
    return (LN_SQRT2PI + log(a) + ((x + 0.5) * log(tmp)) - tmp);
}



float _cl_builtin_lgammaf(float x) {
    x -= 1.0f;
    if (x <= -1.0f)
      return NAN;

    float a = 0.99999999999999709182f;
    float z = x;
    z += 1.0f;
    a += 57.156235665862923517f / z;
    z +=1.0f;
    a += -59.597960355475491248f / z;
    z +=1.0f;
    a += 14.136097974741747174f / z;
    z +=1.0f;
    a += -0.49191381609762019978f / z;
    z +=1.0f;
    a += 0.33994649984811888699e-4f / z;
    z +=1.0f;
    a += 0.46523628927048575665e-4f / z;
    z +=1.0f;
    a += -0.98374475304879564677e-4f / z;
    z +=1.0f;
    a += 0.15808870322491248884e-3f / z;
    z +=1.0f;
    a += -0.21026444172410488319e-3f / z;
    z +=1.0f;
    a += 0.21743961811521264320e-3f / z;
    z +=1.0f;
    a += -0.16431810653676389022e-3f / z;
    z +=1.0f;
    a += 0.84418223983852743293e-4f / z;
    z +=1.0f;
    a += -0.26190838401581408670e-4f / z;
    z +=1.0f;
    a += 0.36899182659531622704e-5f / z;

    float tmp = x + (607/128.0f + 0.5f);
    return (LN_SQRT2PI_F + log(a) + ((x + 0.5f)*log(tmp)) - tmp);
}



IMPLEMENT_EXPR_ALL(lgamma, V_V, _cl_builtin_lgammaf(a), _cl_builtin_lgamma(a))
