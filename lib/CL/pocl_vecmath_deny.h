/* Per-function deny lists for vectorized math libraries.

   Each entry is an LLVM scalar function name as it appears in
   llvm/Analysis/VecFuncs.def (both the libm name and the intrinsic name
   must be listed). A denied function keeps its scalar implementation;
   everything else in the library table is vectorized as before.

   Generated from the ULP harness in the PoCL revitalization sidecar
   (harness/ulp), measured against the OpenCL C 3.0 ULP bounds. Provenance
   is recorded per table. Regenerate when the library or LLVM changes. */

#ifndef POCL_VECMATH_DENY_H
#define POCL_VECMATH_DENY_H

/* glibc libmvec. Measured: glibc 2.39, x86-64 AVX2, LLVM 22.1.8,
   2026-09-04. float log: 3.009 ULP vs 3 ULP bound (CTS width 1 confirms). */
static const char *const PoclVecMathDenyLibmvec[] = {
    "logf", "llvm.log.f32",
};

/* SLEEF GNU-ABI build used through the libmvec table. Measured: SLEEF
   3.5.1, x86-64 AVX2, LLVM 22.1.8, 2026-09-04. double pow: pow(-DBL_MAX, 1)
   returns -inf (SLEEF issue #600, fixed in SLEEF 3.8). Nothing else.
   HOST_CPU_VECMATH_SLEEF_VERSION_NUM is 0 when the version is unknown. */
static const char *const PoclVecMathDenySleef[] = {
#if HOST_CPU_VECMATH_SLEEF_VERSION_NUM < 30800
    "pow", "llvm.pow.f64",
#endif
    /* keep the array non-empty on any version */
    "__pocl_vecmath_none",
};

#endif
