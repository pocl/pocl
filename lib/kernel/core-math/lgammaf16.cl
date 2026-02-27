/* Correctly-rounded logarithm of the absolute value of the gamma function for binary16 value.

Copyright (c) 2023-2025 Alexei Sibidanov and Paul Zimmermann

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

/* This code is based on the binary32 code lgammaf.c:
   at input we convert the inputs to half (exactly),
   we then use the same code than for lgammaf,
   and at output we round to half.
   The changes with respect to lgammaf.c are marked with a comment
   "specific lgammaf16". */

#include "common_types.h"

static double as_r7(double x, constant double *c){
  return (((x-c[0])*(x-c[1]))*((x-c[2])*(x-c[3])))*(((x-c[4])*(x-c[5]))*((x-c[6])));
}

static double as_r8(double x, constant double *c){
  return (((x-c[0])*(x-c[1]))*((x-c[2])*(x-c[3])))*(((x-c[4])*(x-c[5]))*((x-c[6])*(x-c[7])));
}

static double as_sinpi(double x){
  static constant double c[] =
    {0x1p+2, -0x1.de9e64df22ea4p+1, 0x1.472be122401f8p+0, -0x1.d4fcd82df91bp-3,
     0x1.9f05c97e0aab2p-6, -0x1.f3091c427b611p-10, 0x1.b22c9bfdca547p-14, -0x1.15484325ef569p-18};
  x -= 0.5;
  double x2 = x*x, x4 = x2*x2, x8 = x4*x4;
  return (0.25 - x2)*((c[0] + x2*c[1]) + x4*(c[2] + x2*c[3]) + x8*((c[4] + x2*c[5]) + x4*(c[6] + x2*c[7])));
}

static double as_ln(double x){
  b64u64_u t = {.f = x};
  int e = (t.u >> 52) - 0x3ff;
  static constant double c[] =
    {0x1.fffffffffff24p-1, -0x1.ffffffffd1d67p-2, 0x1.55555537802dep-2, -0x1.ffffeca81b866p-3,
     0x1.999611761d772p-3, -0x1.54f3e581b61bfp-3, 0x1.1e642b4cb5143p-3, -0x1.9115a5af1e1edp-4};
  static constant double il[] =
    {0x1.59caeec280116p-57, 0x1.f0a30c01162aap-5, 0x1.e27076e2af2ebp-4, 0x1.5ff3070a793d6p-3,
     0x1.c8ff7c79a9a2p-3, 0x1.1675cababa60fp-2, 0x1.4618bc21c5ec2p-2, 0x1.739d7f6bbd007p-2,
     0x1.9f323ecbf984dp-2, 0x1.c8ff7c79a9a21p-2, 0x1.f128f5faf06ecp-2, 0x1.0be72e4252a83p-1,
     0x1.1e85f5e7040d1p-1, 0x1.307d7334f10bep-1, 0x1.41d8fe84672afp-1, 0x1.52a2d265bc5abp-1};
  static constant double ix[] =
    {0x1p+0, 0x1.e1e1e1e1e1e1ep-1, 0x1.c71c71c71c71cp-1, 0x1.af286bca1af28p-1,
     0x1.999999999999ap-1, 0x1.8618618618618p-1, 0x1.745d1745d1746p-1, 0x1.642c8590b2164p-1,
     0x1.5555555555555p-1, 0x1.47ae147ae147bp-1, 0x1.3b13b13b13b14p-1, 0x1.2f684bda12f68p-1,
     0x1.2492492492492p-1, 0x1.1a7b9611a7b96p-1, 0x1.1111111111111p-1, 0x1.0842108421084p-1};
  int i = (t.u>>48)&0xf;
  t.u = (t.u&(~(ulong)0>>12))|((long)0x3ff<<52);
  double z = ix[i]*t.f - 1, z2 = z*z, z4 = z2*z2;
  return e*0x1.62e42fefa39efp-1 + il[i] + z*((c[0] + z*c[1]) + z2*(c[2] + z*c[3]) + z4*((c[4] + z*c[5]) + z2*(c[6] + z*c[7])));
}

// specific lgammaf16: input renamed to xf16
_CL_OVERLOADABLE half lgamma_r (half xf16, private int *signp) {
  float x = xf16;
  // specific lgammaf16: removed array tb[] of exceptional cases

  float fx = floor(x), ax = fabs(x);
  b32u32_u t = {.f = ax};
  if(__builtin_expect(t.u>=(0xffu<<23), 0)){ // NaN or Inf
    if(t.u==(0xffu<<23)){ // +-inf
      *signp = 1;
      return 1.0f/0.0f;
    }
    return x + x; // nan
  }
  if(__builtin_expect(fx==x, 0)){ // x integer
    if(x <= 0.0f) {
      t.f = x;
      // gamma(+0) = +Inf, gamma(-0) = -Inf
      if(!(t.u<<1)) *signp = 1 - 2*(t.u >> 31);
      return 1.0f/0.0f;
    }
    if(x==1.0f || x==2.0f) {
      *signp = 1;
      return 0.0f;
    }
  }

  double s = x;
  /* Check the value of fx to avoid a spurious invalid exception.
     Note that for a binary32 |x| >= 2^23, x is necessarily an integer,
     and we already dealed with negative integers, thus now:
     -2^23 < x < +Inf and x is not a negative integer nor 0, 1, 2. */
  if (__builtin_expect (fx >= 0, 1))
    *signp = 1;
  else
    // gamma(x) is negative in (-2n-1,-2n), thus when fx is odd
    *signp = 1 - ((((int)fx)&1)<<1);
  double z = ax, f;
  if(__builtin_expect(ax<0x1.52p-1f, 0)){
    static constant double rn[] =
      {-0x1.505bdf4b65acp+4, -0x1.51c80eb47e068p+2, 0x1.0000000007cb8p+0, -0x1.4ac529250a1fcp+1,
       -0x1.a8c99dbe1621ap+0, -0x1.4abdcc74115eap+0, -0x1.1b87fe5a5b923p+0, -0x1.05b8a4d47ff64p+0};
    const double c0 = 0x1.0fc0fad268c4dp+2;
    static constant double rd[] =
      {-0x1.4db2cfe9a5265p+5, -0x1.062e99d1c4f27p+3, -0x1.c81bc2ecf25f6p+1, -0x1.108e55c10091bp+1,
       -0x1.7dd25af0b83d4p+0, -0x1.36bf1880125fcp+0, -0x1.1379fc8023d9cp+0, -0x1.03712e41525d2p+0};
    f = (c0*s)*as_r8(s, rn)/as_r8(s, rd) - as_ln(z);
  } else { // |x| >= 0x1.52p-1
    if(ax > 0x1.afc1ap+1f){
      if(__builtin_expect(x >= 0x1.ff4p+12f, 0)){ // specific lgammaf16
        /* for x=0x1.ff4p+12, lgamma(x) ~ 0x1.ffd3p+15, thus there is no
           overflow for rounding towards zero, downwards or to nearest */
#ifndef __clang__
	half r = (x > 0x1.ff4p+12f) ? 0x1p15f16 * 0x1p15f16
          : 0x1.ffcp+15f16 + 1.0f16;
#else
        /* clang 19 raises a spurious overflow with the above code
           for x=0x1.ff4p+12 and RNDN with -frounding-math: apparently it
           evaluates the ? part even if the conditional is false:
           https://github.com/llvm/llvm-project/issues/157395.
           We thus use a workaround. */
        half r;
        if (x > 0x1.ff4p+12f)
          r = 0x1p15f16 * 0x1p15f16;
        else
          r = 0x1.ffcp+15f16 + 1.0f16;
#endif
#ifdef CORE_MATH_SUPPORT_ERRNO
	if (x > 0x1.ff4p+12f || (x == 0x1.ff4p+12f && x + 0x1p-12f > x))
          errno = ERANGE; // overflow
#endif
	return r;
      }
      double lz = as_ln(z);
      f = (z - 0.5)*(lz - 1) + 0x1.acfe390c97d69p-2;
      // specific lgammaf16: remove the condition ax < 0x1.0p+20f (always true)
      double iz = 1.0/z, iz2 = iz*iz;
      if(ax > 1198.0f){
        f +=  iz*(1./12.);
      } else if(ax > 0x1.279a7p+6f){
        static constant double c[] =
          {0x1.555555547fbadp-4, -0x1.6c0fd270c465p-9};
        f +=  iz*(c[0] + iz2*c[1]);
      } else if(ax > 0x1.555556p+3f){
        static constant double c[] =
          {0x1.555555554de0bp-4, -0x1.6c16bdc45944fp-9, 0x1.a0077f300ecb3p-11, -0x1.2e9cfff3b29c2p-11};
        double iz4 = iz2*iz2;
        f +=  iz*((c[0] + iz2*c[1]) + iz4*(c[2] + iz2*c[3]));
      } else {
        static constant double c[] =
          {0x1.5555555551286p-4, -0x1.6c16c0e7c4cf4p-9, 0x1.a0193267fe6f2p-11, -0x1.37e87ec19cb45p-11,
           0x1.b40011dfff081p-11, -0x1.c16c8946b19b6p-10, 0x1.e9f47ace150d8p-9, -0x1.4f5843a71a338p-8};
        double iz4 = iz2*iz2, iz8 = iz4*iz4;
        double p = ((c[0] + iz2*c[1]) + iz4*(c[2] + iz2*c[3])) + iz8*((c[4] + iz2*c[5]) + iz4*(c[6] + iz2*c[7]));
        f += iz*p;
      }
      if(x < 0.0f){
	f = 0x1.250d048e7a1bdp+0 - f - lz;
	double lp = as_ln(as_sinpi(x-fx));
	f -= lp;
      }
    } else {
      static constant double rn[] =
	{-0x1.667923ff14df7p+5, -0x1.2d35f25ad8f64p+3, -0x1.b8c9eab9d5bd3p+1, -0x1.7a4a97f494127p+0,
	 -0x1.3a6c8295b4445p-1, -0x1.da44e8b810024p-3, -0x1.9061e81c77e4ap-5};
      const double c0 = 0x1.3cc0e6a0106b3p+2;
      static constant double rd[] =
	{-0x1.491a899e84c52p+6, -0x1.d202961b9e098p+3, -0x1.4ced68c631ed6p+2, -0x1.2589eedf40738p+1,
	 -0x1.1302e3337271p+0, -0x1.c36b802f26dffp-2, -0x1.3ded448acc39dp-3, -0x1.bffc491078eafp-6};
      f = (z-1)*(z-2)*c0*as_r7(z, rn)/as_r8(z, rd);
      if(x < 0.0f){
	if(__builtin_expect(t.u < 0x40301b93u && t.u > 0x402f95c2u, 0)){
	  double h = (s + 0x1.5fb410a1bd901p+1) - 0x1.a19a96d2e6f85p-54, h2 = h*h, h4 = h2*h2;
	  static constant double c[] =
	    {-0x1.ea12da904b18cp+0, 0x1.3267f3c265a54p+3, -0x1.4185ac30cadb3p+4, 0x1.f504accc3f2e4p+5,
	     -0x1.8588444c679b4p+7, 0x1.43740491dc22p+9, -0x1.12400ea23f9e6p+11, 0x1.dac829f365795p+12};
	  f = h*((c[0] + h*c[1]) + h2*(c[2] + h*c[3]) + h4*((c[4] + h*c[5]) + h2*(c[6] + h*c[7])));
	} else if(__builtin_expect(t.u >  0x401ceccbu && t.u < 0x401d95cau, 0)){
	  double h = (s + 0x1.3a7fc9600f86cp+1) + 0x1.55f64f98af8dp-55, h2 = h*h, h4 = h2*h2;
	  static constant double c[] =
	    {0x1.83fe966af535fp+0, 0x1.36eebb002f61ap+2, 0x1.694a60589a0b3p+0, 0x1.1718d7aedb0b5p+3,
	     0x1.733a045eca0d3p+2, 0x1.8d4297421205bp+4, 0x1.7feea5fb29965p+4};
	  f = h*((c[0] + h*c[1]) + h2*(c[2] + h*c[3]) + h4*((c[4] + h*c[5]) + h2*(c[6])));
#ifdef CORE_MATH_SUPPORT_ERRNO // specific lgammaf16
    if (x == -0x1.3a8p+1)
      errno = ERANGE; // underflow
#endif
	} else if(__builtin_expect(t.u > 0x40492009u && t.u < 0x404940efu, 0)){
	  double h = (s + 0x1.9260dbc9e59afp+1) + 0x1.f717cd335a7b3p-53, h2 = h*h, h4 = h2*h2;
	  static constant double c[] =
	    {0x1.f20a65f2fac55p+2, 0x1.9d4d297715105p+4, 0x1.c1137124d5b21p+6, 0x1.267203d24de38p+9,
	     0x1.99a63399a0b44p+11, 0x1.2941214faaf0cp+14, 0x1.bb912c0c9cdd1p+16};
	  f = h*((c[0] + h*c[1]) + h2*(c[2] + h*c[3]) + h4*((c[4] + h*c[5]) + h2*(c[6])));
	} else {
	  f = 0x1.250d048e7a1bdp+0 - f;
	  double lp = as_ln(as_sinpi(x-fx)*z);
	  f -= lp;
	}
      }
    }
  }

  half r = convert_half(f); // specific lgammaf16
  // specific lgammaf16: removed exceptional cases
  return r;
}

_CL_OVERLOADABLE half lgamma_r (half xf16, local int *signp) {
    int sign = 0;
    half retval = lgamma_r(xf16, &sign);
    *signp = sign;
    return retval;
}

_CL_OVERLOADABLE half lgamma_r (half xf16, global int *signp) {
    int sign = 0;
    half retval = lgamma_r(xf16, &sign);
    *signp = sign;
    return retval;
}

_CL_OVERLOADABLE half lgamma_r (half xf16, generic int *signp) {
    int sign = 0;
    half retval = lgamma_r(xf16, &sign);
    *signp = sign;
    return retval;
}

_CL_OVERLOADABLE half lgamma (half xf16) {
    int sign = 0;
    return lgamma_r(xf16, &sign);
}
