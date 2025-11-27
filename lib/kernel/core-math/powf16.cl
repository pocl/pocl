/* Correctly-rounded power function for binary16 value.

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

#if 0
#include <fenv.h> // for fexcept_t, fegetexceptflag, FE_INEXACT
#define FLAG_T fexcept_t
#endif

static constant b16u16_u poszero = {.u = 0x0000};
static constant b16u16_u negzero = {.u = 0x8000};
static constant b16u16_u one = {.u = 0x3c00};
static constant b16u16_u neginf = {.u = 0xfc00};
static constant b16u16_u posinf = {.u = 0x7c00};

static inline int isint(b16u16_u v) {
	if (v.f == 0.0f) return 1;
	return (v.u & 0x7fff) >> 10 >= 25 - __builtin_ctz(0x400 | v.u);
}

static inline int isodd(b16u16_u v) {
	if (v.f == 0.0f) return 0;
	return (v.u & 0x7fff) >> 10 == 25 - __builtin_ctz(0x400 | v.u);
}

// return non-zero if x^y might be representable with 11 precision bits
// See https://ens-lyon.hal.science/ensl-00169409v2/file/power-rr.lip.pdf
// for more details
static inline ulong is_exact(b16u16_u x, b16u16_u y) {
	// S_1 = {(x, y) | x = 2^E and -25 <= y*E <= 15}
	// S_2 = {(x, y) | y = n2^F with 1 <= n <= 7 and -2 <= F < 0}
		// ie y = 1/4, 1/2, 3/4, 5/4, 3/2, 7/4, 5/2, 7/2
	// S_3 = {(x, y) | 1 <= y <= 7 (integer)}

	// S_1
	if (!(x.u & 0x3ff)) { // x = 2^E with E >= -14
		int E = (x.u >> 10) - 15;
		int yE = E * (float) y.f;
		if (((y.u >> 10) & 0x1f) + __builtin_ctz(y.u | 0x400) + __builtin_ctz(E) >= 25 && -25 <= yE && yE <= 15) return (0x3ffull + yE) << 52;
	}
	else if (__builtin_ctz(x.u) + __builtin_clz(x.u) == 31) { // x = 2^E with E < -14
		int E = -24 + __builtin_ctz(x.u);
		int yE = E * (float) y.f;
		if (((y.u >> 10) & 0x1f) + __builtin_ctz(y.u | 0x400) + __builtin_ctz(E) >= 25 && -25 <= yE && yE <= 15) return (0x3ffull + yE) << 52;
	}
	// S_2 and S_3
	if (!(y.u & 0xff)) { // return (n << 4) + 2^-F
		if (y.u == 0x3400) return 0x14; // 0.25
		if (y.u == 0x3800) return 0x12; // 0.5
		if (y.u == 0x3a00) return 0x34; // 0.75
		if (y.u == 0x3c00) return 0x11; // 1
		if (y.u == 0x3d00) return 0x54; // 1.25
		if (y.u == 0x3e00) return 0x32; // 1.5
		if (y.u == 0x3f00) return 0x74; // 1.75
		if (y.u == 0x4000) return 0x21; // 2
		if (y.u == 0x4100) return 0x52; // 2.5
		if (y.u == 0x4200) return 0x31; // 3
		if (y.u == 0x4300) return 0x72; // 3.5
		if (y.u == 0x4400) return 0x41; // 4
		if (y.u == 0x4500) return 0x51; // 5
		if (y.u == 0x4600) return 0x61; // 6
		if (y.u == 0x4700) return 0x71; // 7
	}
	return 0;
}

#if 0
// don't use the MXCSR register since it is not affected by half operations
static FLAG_T get_flag (void) {
  fexcept_t flag;
  fegetexceptflag (&flag, FE_INEXACT);
  return flag;
}

static void set_flag (FLAG_T flag) {
  fesetexceptflag (&flag, FE_INEXACT);
}
#endif

static inline double fast_pow(double x, ulong y) {
	double ret = 1;
	if (y & 4) ret *= x;
	ret *= ret;
	if (y & 2) ret *= x;
	ret *= ret;
	if (y & 1) ret *= x;
	return ret;
}

static inline double log_in_pow(double x) {
	b64u64_u xd = {.f = x};
	static constant double log2 = 0x1.62e42fefa39efp-1;
	static constant double tb[] = // tabulate value of log(1 + i2^-6) for i in [0, 63]
		{0x0p+0, 0x3.f815161f807c8p-8, 0x7.e0a6c39e0ccp-8, 0xb.ba2c7b196e7ep-8,
		 0xf.85186008b153p-8, 0x1.341d7961bd1d1p-4, 0x1.6f0d28ae56b4cp-4, 0x1.a926d3a4ad563p-4,
		 0x1.e27076e2af2e6p-4, 0x2.1aefcf9a11cb2p-4, 0x2.52aa5f03fea46p-4, 0x2.89a56d996fa3cp-4,
		 0x2.bfe60e14f27a8p-4, 0x2.f57120421b212p-4, 0x3.2a4b539e8ad68p-4, 0x3.5e7929d017fe6p-4,
		 0x3.91fef8f353444p-4, 0x3.c4e0edc55e5ccp-4, 0x3.f7230dabc7c56p-4, 0x4.28c9389ce438cp-4,
		 0x4.59d72aeae9838p-4, 0x4.8a507ef3de598p-4, 0x4.ba38aeb8474c4p-4, 0x4.e993155a517a8p-4,
		 0x5.1862f08717b08p-4, 0x5.46ab61cb7e0b4p-4, 0x5.746f6fd602728p-4, 0x5.a1b207a6c52bcp-4,
		 0x5.ce75fdaef401cp-4, 0x5.fabe0ee0abf0cp-4, 0x6.268ce1b05096cp-4, 0x6.51e5070845becp-4,
		 0x6.7cc8fb2fe613p-4, 0x6.a73b26a682128p-4, 0x6.d13ddef323d8cp-4, 0x6.fad36769c6dfp-4,
		 0x7.23fdf1e6a6888p-4, 0x7.4cbf9f803af54p-4, 0x7.751a813071284p-4, 0x7.9d109875a1e2p-4,
		 0x7.c4a3d7ebc1bb4p-4, 0x7.ebd623de3cc7cp-4, 0x8.12a952d2e87f8p-4, 0x8.391f2e0e6fap-4,
		 0x8.5f39721295418p-4, 0x8.84f9cf16a64b8p-4, 0x8.aa61e97a6af5p-4, 0x8.cf735a33e4b78p-4,
		 0x8.f42faf382068p-4, 0x9.18986bdf5fa18p-4, 0x9.3caf0944d88d8p-4, 0x9.6074f6a24746p-4,
		 0x9.83eb99a7885fp-4, 0x9.a7144ece70e98p-4, 0x9.c9f069ab150dp-4, 0x9.ec813538ab7d8p-4,
		 0xa.0ec7f4233957p-4, 0xa.30c5e10e2f61p-4, 0xa.527c2ed81f5d8p-4, 0xa.73ec08dbadd88p-4,
		 0xa.9516932de2d58p-4, 0xa.b5fcead9f9cc8p-4, 0xa.d6a0261acf968p-4, 0xa.f70154920b3a8p-4};
	static constant double tl[] = // tabulate value of 1 / (1 + i2^-6) for i in [0, 63]
		{0x1p-52, 0xf.c0fc0fc0fc1p-56, 0xf.83e0f83e0f84p-56, 0xf.4898d5f85bb38p-56,
		 0xf.0f0f0f0f0f0fp-56, 0xe.d7303b5cc0ed8p-56, 0xe.a0ea0ea0ea0e8p-56, 0xe.6c2b4481cd858p-56,
		 0xe.38e38e38e38ep-56, 0xe.070381c0e07p-56, 0xd.d67c8a60dd68p-56, 0xd.a740da740da78p-56,
		 0xd.79435e50d794p-56, 0xd.4c77b03531dfp-56, 0xd.20d20d20d20dp-56, 0xc.f6474a8819ec8p-56,
		 0xc.cccccccccccdp-56, 0xc.a4587e6b74fp-56, 0xc.7ce0c7ce0c7dp-56, 0xc.565c87b5f9d5p-56,
		 0xc.30c30c30c30cp-56, 0xc.0c0c0c0c0c0cp-56, 0xb.e82fa0be82fap-56, 0xb.c52640bc5264p-56,
		 0xb.a2e8ba2e8ba3p-56, 0xb.81702e05c0b8p-56, 0xb.60b60b60b60b8p-56, 0xb.40b40b40b40b8p-56,
		 0xb.21642c8590b2p-56, 0xb.02c0b02c0b03p-56, 0xa.e4c415c9882b8p-56, 0xa.c7691840ac768p-56,
		 0xa.aaaaaaaaaaaa8p-56, 0xa.8e83f5717c0a8p-56, 0xa.72f05397829c8p-56, 0xa.57eb50295fad8p-56,
		 0xa.3d70a3d70a3d8p-56, 0xa.237c32b16cfd8p-56, 0xa.0a0a0a0a0a0ap-56, 0x9.f1165e725481p-56,
		 0x9.d89d89d89d8ap-56, 0x9.c09c09c09c0ap-56, 0x9.a90e7d95bc608p-56, 0x9.91f1a515885f8p-56,
		 0x9.7b425ed097b4p-56, 0x9.64fda6c0965p-56, 0x9.4f2094f2094fp-56, 0x9.39a85c40939a8p-56,
		 0x9.249249249249p-56, 0x9.0fdbc090fdbcp-56, 0x8.fb823ee08fb8p-56, 0x8.e78356d1408e8p-56,
		 0x8.d3dcb08d3dcbp-56, 0x8.c08c08c08c09p-56, 0x8.ad8f2fba93868p-56, 0x8.9ae4089ae4088p-56,
		 0x8.8888888888888p-56, 0x8.767ab5f34e48p-56, 0x8.64b8a7de6d1d8p-56, 0x8.5340853408538p-56,
		 0x8.421084210842p-56, 0x8.3126e978d4fep-56, 0x8.208208208208p-56, 0x8.102040810204p-56};
	int expo = (xd.u >> 52) - 1023;
	int i = (xd.u & (0x3full << 46)) >> 46;
	xd.f = (xd.u & 0x3fffffffffff) * tl[i];
	xd.f *= fma(fma(fma(fma(0x1.999999999999ap-3, xd.f, -0.25), xd.f, 0x1.5555555555555p-2), xd.f, -0.5), xd.f, 1.0);
	// xd.f *= fma(fma(0x1.5555555555555p-2, xd.f, -0.5), xd.f, 1.0);
	return fma(log2, (double) expo, tb[i] + xd.f);
}

static inline double exp_in_pow(double x) {
	b64u64_u xd = {.f = x};
	static constant b64u64_u x0 = {.f = -0x1.1542457337d44p+4};
	static constant b64u64_u x1 = {.f = 0x1.62e42fefa39efp+3};
	static constant double tb[] = // tabulate value of exp(log(2)*i/64) for i in [0, 63]
		{0x1p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0, 0x1.0874518759bc8p+0,
		 0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0, 0x1.11301d0125b51p+0, 0x1.1429aaea92dep+0,
		 0x1.172b83c7d517bp+0, 0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
		 0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0, 0x1.2d285a6e4030bp+0,
		 0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0, 0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0,
		 0x1.3dea64c123422p+0, 0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cdp+0,
		 0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0, 0x1.56f4736b527dap+0,
		 0x1.5ab07dd485429p+0, 0x1.5e76f15ad2149p+0, 0x1.6247eb03a5585p+0, 0x1.6623882552225p+0,
		 0x1.6a09e667f3bccp+0, 0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
		 0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0, 0x1.868d99b4492ecp+0,
		 0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0, 0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e4fp+0,
		 0x1.9c49182a3f09p+0, 0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdcp+0,
		 0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0, 0x1.bcc1e904bc1d2p+0,
		 0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0, 0x1.cb720dcef9069p+0, 0x1.d072d4a07897bp+0,
		 0x1.d5818dcfba487p+0, 0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
		 0x1.ea4afa2a490d9p+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4541p+0, 0x1.fa7c1819e90d8p+0};
	if ((xd.u & (0x7ffull << 52)) == (0x7ffull << 52)) { // if x is NaN or x is inf
		if (xd.u == 0xffull << 25) return 0.0; // x is -Inf
		else return xd.f + xd.f;
	}
	else if (xd.u > x0.u) return 0x1p-25;
	else if (xd.f > x1.f) return 0x1.ffcp+15 + 0x1p+5;
	static constant double sixtyfour_over_log2 = 0x1.71547652b82fep+6;
	static constant double minus_log2_over_sixtyfour = -0x1.62e42fefa39efp-7;
	double j = __builtin_roundeven(sixtyfour_over_log2 * xd.f);
	long jint = j;
	int i = jint & 0x3f;
	double xp = fma(minus_log2_over_sixtyfour, j, xd.f);
	xp = fma(fma(fma(0x1.55555555555p-3, xp, 0.5), xp, 1.0), xp, 1.0);
	b64u64_u ret = {.f = xp * tb[i]};
	ret.u += (jint >> 6) * (1ull << 52);
	return ret.f;
}

_CL_OVERLOADABLE half pow(half x, half y) {
#if 0
  volatile FLAG_T flag = get_flag ();
#endif
	b16u16_u vx = {.f = x}, vy = {.f = y};
	ulong sign = 0;
	if ((vx.u & 0x7fff) == 0x3c00) { // |x| = 1
		if (vx.u >> 15) { // x = -1
            if (isnan(y))
                return y; // y = NaN
            if (isint(vy))
                return (isodd(vy)) ? vx.f : -vx.f;
			return NAN;
		}
        // x = +1
        return x;
    }
    if ((vy.u & 0x7fff) == 0) { // y = 0
        return one.f;
    }
	// x^0 = 1 except if x = sNaN
	if ((vy.u & 0x7fff) >= 0x7c00) { // y = Inf/NaN
		// the case |x| = 1 was checked above
        if ((vx.u & 0x7fff) > 0x7c00) return NAN; // x = NaN
		if ((vy.u & 0x7fff) == 0x7c00) { // y = +/-Inf
            if (((vx.u & 0x7fff) < 0x3c00) == (vy.u >> 15)) {
				return posinf.f; // |x| < 1 && y = -Inf or |x| > 1 && y = +Inf
			} else {
				return poszero.f; // |x| < 1 && y = +Inf or |x| > 1 && y = -Inf
			}
		}
		return NAN; // y = NaN
	}
	if (!(vx.u & 0x7fff)) { // if x = 0
		if (vy.u >> 15) { // y < 0
		  if (isodd(vy) && vx.u >> 1)
			  return neginf.f;
			else
			  return posinf.f;
		} else { // y > 0
		  if (isodd(vy) && vx.u >> 1)
			  return negzero.f;
			else
			  return poszero.f;
		}
	}
	if (vx.u >= 0x7c00) { // x = Inf or x = NaN or x <= 0
		if ((vx.u & 0x7fff) == 0x7c00) { // x = +/-Inf
			if (!isodd(vy)) vx.u &= 0x7fff; // y even -> ret will be positive
			if (vy.u >> 15) vx.u &= 0x8000; // y < 0 -> ret will be +/-0
			return vx.f;
		}
		if ((vx.u & 0x7fff) > 0x7c00) return x + x; // x is NaN
		// x < 0
		if (!isint(vy)) {
			return NAN;
		} 
		else if (isodd(vy)) sign = 1ull << 63;
		vx.u &= 0x7fff;
	}
	// some wrong cases
	if (vx.u == 0x1c94 && vy.u == 0x31bc) return 0x1.848p-2f - 0x1p-14f;
	if (vx.u == 0x537b && vy.u == 0x25bf) return 0x1.18cp+0f - 0x1p-12f;
	if (vx.u == 0x756b && vy.u == 0x112e) return 0x1.01cp+0f - 0x1p-12f;
	if (vx.u == 0x0d36 && vy.u == 0x2316) return 0x1.cap-1f + 0x1p-13f;
	if (vx.u == 0x273b && vy.u == 0x38b3) return 0x1.f8p-4f + 0x1p-16f;
	if (vx.u == 0x32bb && vy.u == 0x4242) return 0x1.f2cp-8f + 0x1p-20f;
	if (vx.u == 0x4d47 && vy.u == 0x9d5f) return 0x1.f8p-1 - 0x1p-13;
	if (vx.u == 0x2e27 && vy.u == 0xc107) return 0x1.688p+8 - 0x1p-4;
	if (vx.u == 0x14cb && vy.u == 0xbe46) return 0x1.35cp+15 - 0x1p-3;
	if (vx.u == 0x7abf && vy.u == 0x8cc5) return 0x1.fe8p-1 - 0x1p-13;
	if (vx.u == 0x650c && vy.u == 0x9bed) return 0x1.f2p-1 + 0x1p-13;
	if (vx.u == 0x29a0 && vy.u == 0xb5cf) return 0x1.8ep+1 + 0x1p-11;
	if (vx.u == 0x17e9 && vy.u == 0xb1cf) return 0x1.8ep+1 + 0x1p-11;
	if (vx.u == 0x5988 && vy.u == 0x9443) return 0x1.fd4p-1 + 0x1p-13;
	ulong isex = is_exact(vx, vy);
	b64u64_u ret;
	if (isex > 0xff) ret.u = isex;
	else if (isex) {
          ret.f = exp_in_pow(log_in_pow(vx.f) * (double) vy.f);
		// Test if ret is exact :
		b64u64_u test_exact1 = {.f = fast_pow(vx.f, isex >> 4)};
		b64u64_u test_exact2 = {.u = (ret.u + (0x1ull << 40)) & (0xfffffeull << 40)};
		test_exact2.f = fast_pow(test_exact2.f, isex & 0xf);
		if (test_exact1.u == test_exact2.u) {
			ret.u = (ret.u + (1ull << 40)) & (0xfffffeull << 40);
#if 0
			set_flag(flag); // resetting inexact (if inexact -> raise in return)
#endif
		}
	}
    else
      ret.f = exp_in_pow(log_in_pow(vx.f) * (double) vy.f);
	ret.u += sign;
    return convert_half(ret.f);
}


/*
ERROR: pown: inf ulp error at {-0x1.5bcp-9 (0x996f), -1999335705}
Expected: -inf (half 0xfc00)
Actual: inf (half 0x7c00) at index: 4

ERROR: pown: inf ulp error at {-0x1.fbcp+12 (0xefef), 1657969807}
Expected: -inf (half 0xfc00)
Actual: inf (half 0x7c00) at index: 9

ERROR: pown: inf ulp error at {-0x1.f88p+8 (0xdfe2), 958472645}
Expected: -inf (half 0xfc00)
Actual: inf (half 0x7c00) at index: 9
*/

_CL_OVERLOADABLE half pown(half x, int y) {
    float ret = pow(convert_float(x), y);
    return convert_half(ret);
}

/*

ERROR: powr: nan ulp error at {-0x1.be8p-12 (0x8efa), 0x1.becp+12 (0x6efb)}
Expected: nan  (half 0x7e00)
Actual: 0x0p+0 (half 0x0000) at index: 0

ERROR: powr: nan ulp error at {-0x1.728p+9 (0xe1ca), -0x1.4ap+14 (0xf528)}
Expected: nan  (half 0x7e00)
Actual: 0x0p+0 (half 0x0000) at index: 14

ERROR: powr: nan ulp error at {-0x1.5dcp-10 (0x9577), -0x1.c84p+12 (0xef21)}
Expected: nan  (half 0x7e00)
Actual: inf (half 0x7c00) at index: 9

ERROR: powr: nan ulp error at {-0x1.f24p+13 (0xf3c9), -0x1.e84p+10 (0xe7a1)}
Expected: nan  (half 0x7e00)
Actual: -0x0p+0 (half 0x8000) at index: 7

*/

_CL_OVERLOADABLE half powr(half x, half y) {
    b16u16_u xx = { .f = x };
    x = (xx.u == 0x8000) ? 0 : x;
    half retval = pow(x, y);
    retval = (xx.u > 0x8000) ? NAN : retval;
    retval = isnan(x) ? NAN : retval;
    retval = isnan(y) ? NAN : retval;
    return retval;
}

