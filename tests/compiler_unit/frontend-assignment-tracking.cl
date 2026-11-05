// The front-end runs Clang's assignment tracking on optimized builds with
// debug info, so the values of variables whose address escapes are tracked
// through the stores to them.
__attribute__((noinline)) void escape(private int *p) {
  *(volatile private int *)p += 1;
}

kernel void track(global int *out, int n) {
  int x = n;
  escape(&x);
  x += 5;
  escape(&x);
  out[0] = x;
}
