/* A scalar-typed kernel reaches the vector math library through the loop
   vectorizer: the work-item loop is vectorized and the swapped builtin's
   scalar call is mapped to the library's vector variant (libmvec ABI,
   _ZGV<isa>N<lanes>v_sinf; the ISA letter and lane count depend on the
   host). Requires ENABLE_HOST_CPU_VECTORIZE_LIBMVEC with glibc libmvec or
   SLEEF's gnuabi build. */
kernel void loopvec_libcall(global const float* a, global float* b)
{
  size_t i = get_global_id(0);
  b[i] = sin(a[i]);
}
