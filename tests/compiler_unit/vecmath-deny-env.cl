/* The same kernel as vecmath-loopvec-libcall.cl, run with
   POCL_VECMATH_DENY=sinf,llvm.sin.f32: the denied function must not be
   mapped to a vector library variant. The loop still vectorizes on the
   LLVM intrinsic, which the backend later lowers to scalar libm calls. */
kernel void deny_env(global const float* a, global float* b)
{
  size_t i = get_global_id(0);
  b[i] = sin(a[i]);
}
