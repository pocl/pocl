// The front-end marks plain loops with interleave count 1, which keeps the
// loop vectorizer from interleaving them as Clang's -fno-unroll-loops used
// to, and leaves loops with explicit hints alone.
kernel void hints(global int *out, global const int *in, int n) {
  size_t i = get_global_id(0);
  int a = in[i], b = a, c = a;
  for (int k = 0; k < n; ++k)
    a = a * in[k] + 1;
  #pragma nounroll
  for (int k = 0; k < n; ++k)
    b = b * in[k] + 2;
  #pragma clang loop interleave_count(2)
  for (int k = 0; k < n; ++k)
    c = c * in[k] + 3;
  out[i] = a + b + c;
}
