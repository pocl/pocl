/* Per-function deny lists for vectorized math libraries.

   Each entry is an LLVM scalar function name as it appears in
   llvm/Analysis/VecFuncs.def (both the libm name and the intrinsic name
   must be listed). A denied function keeps its scalar implementation;
   everything else in the library table is vectorized as before.

   Each entry was measured against the OpenCL C 3.0 ULP bounds; the
   measurement is recorded beside each table. Re-measure when the library
   or LLVM changes. */

#ifndef POCL_VECMATH_DENY_H
#define POCL_VECMATH_DENY_H

/* glibc libmvec. Measured: glibc 2.39, x86-64 AVX2, LLVM 22.1.8.
   float log: 3.009 ULP vs 3 ULP bound (2026-09-04, CTS width 1 confirms).
   double exp: 3.006 ULP vs 3 ULP bound (2026-09-24, CTS math_exp fp64, at
   -0x1.e8000000001c2p-9). */
static const char *const PoclVecMathDenyLibmvec[] = {
    "logf", "llvm.log.f32",
    "exp", "llvm.exp.f64",
};

/* SLEEF GNU-ABI build used through the libmvec table. Measured: SLEEF
   3.5.1, x86-64 AVX2, LLVM 22.1.8, 2026-09-04. double pow: pow(-DBL_MAX, 1)
   returns -inf (SLEEF issue #600, fixed in SLEEF 3.8). Nothing else. */
static const char *const PoclVecMathDenySleef[] = {
    "pow", "llvm.pow.f64",
};

#endif
