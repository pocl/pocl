/* OpenCL built-in library: erf()

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

/*
 * The following code is adapted from
 * http://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
 * which is licensed Public Domain.
 *
 * severely limited precision, only useful for floats.
 * TODO it's probably not accurate enough on the entire range.
 */

float _cl_builtin_erff(float g)
{
  float x = fabs(g);
  if (x >= 4.0f)
    return (g > 0.0f) ? 1.0f : -1.0f;

  // constants
  float a1 =  0.254829592f;
  float a2 = -0.284496736f;
  float a3 =  1.421413741f;
  float a4 = -1.453152027f;
  float a5 =  1.061405429f;
  float p  =  0.3275911f;

  // A&S formula 7.1.26
  float t = 1.0f / fma(p, x, 1.0f);
  float temp = fma(a5, t, a4);
  temp = fma(temp, t, a3);
  temp = fma(temp, t, a2);
  temp = fma(temp, t, a1);
  temp *= t;
  float y = fma(temp, exp(-x*x), -1.0f);

  return (g > 0.0f) ? -y : y;
}

float _cl_builtin_erfcf(float g);





/* The following code is adapted from
 * from cpython/Modules/mathmodule.c
 *
 * Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2011, 2012, 2013, 2014, 2015 Python Software Foundation; All Rights Reserved
 * which has Python Software License (BSD compatible).
 *
 * Changes:
 * - adapt to OpenCL builtins (isnan etc)
 * - use fma() where possible.
 */


#define ERF_SERIES_CUTOFF 1.5
#define ERF_SERIES_TERMS 25
#define ERFC_CONTFRAC_CUTOFF 30.0
#define ERFC_CONTFRAC_TERMS 50

#define SQRTPI 1.772453850905516027298167483341145182798

/*
   Error function, via power series.
   Given a finite float x, return an approximation to erf(x).
   Converges reasonably fast for small x.
*/

double m_erf_series(double x)
{
    double x2, acc, fk, result, temp;
    int i;

    x2 = x * x;
    acc = 0.0;
    fk = (double)ERF_SERIES_TERMS + 0.5;
    for (i = 0; i < ERF_SERIES_TERMS; i++) {
        temp = acc / fk;
        //acc = 2.0 + x2 * temp;
        acc = fma(x2, temp, 2.0);
        fk -= 1.0;
    }
    return (acc * x * exp(-x2) / SQRTPI);
}

/*
   Complementary error function, via continued fraction expansion.
   Given a positive float x, return an approximation to erfc(x).  Converges
   reasonably fast for x large (say, x > 2.0), and should be safe from
   overflow if x and nterms are not too large.  On an IEEE 754 machine, with x
   <= 30.0, we're safe up to nterms = 100.  For x >= 30.0, erfc(x) is smaller
   than the smallest representable nonzero float.  */

double m_erfc_contfrac(double x)
{
    double x2, a, da, p, p_last, q, q_last, b, result;
    int i;

    if (x >= ERFC_CONTFRAC_CUTOFF)
        return 0.0;

    x2 = x*x;
    a = 0.0;
    da = 0.5;
    p = 1.0; p_last = 0.0;
    q = da + x2; q_last = 1.0;
    for (i = 0; i < ERFC_CONTFRAC_TERMS; i++) {
        double temp;
        a += da;
        da += 2.0;
        b = da + x2;
        //temp = p; p = b*p - a*p_last; p_last = temp;
        temp = p; p = fma(b, p, -a*p_last); p_last = temp;
        //temp = q; q = b*q - a*q_last; q_last = temp;
        temp = q; q = fma(b, q, -a*q_last); q_last = temp;
    }
    return (p / q * x * exp(-x2) / SQRTPI);
}

/* Error function erf(x), for general x */

double _cl_builtin_erf(double x)
{
    double absx, cf;

    if (isnan(x))
        return x;
    absx = fabs(x);
    if (absx < ERF_SERIES_CUTOFF)
        return m_erf_series(x);
    else {
        cf = m_erfc_contfrac(absx);
        return (x > 0.0) ? 1.0 - cf : cf - 1.0;
    }
}

/* Complementary error function erfc(x), for general x. */

double _cl_builtin_erfc(double x)
{
    double absx, cf;

    if (isnan(x))
        return x;
    absx = fabs(x);
    if (absx < ERF_SERIES_CUTOFF)
        return 1.0 - m_erf_series(x);
    else {
        cf = m_erfc_contfrac(absx);
        return (x > 0.0) ? cf : 2.0 - cf;
    }
}


IMPLEMENT_EXPR_ALL(erf, V_V, _cl_builtin_erff(a), _cl_builtin_erf(a))

IMPLEMENT_EXPR_ALL(erfc, V_V, _cl_builtin_erfcf(a), _cl_builtin_erfc(a))
