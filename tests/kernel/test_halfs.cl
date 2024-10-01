#pragma OPENCL EXTENSION cl_khr_fp16 : enable

global half a = INFINITY;
global half b = 1.0h;
global half8 va = (half8)(INFINITY);
global half8 vb = (half8)(1.0h);

/* This prevents compiler to optimize away test inputs without volatile keyword.
 */
kernel void touch_testdata(half a_init, half b_init, half va_init,
                           half vb_init) {
  a = a_init;
  b = b_init;
  va = va_init;
  vb = vb_init;
}

kernel
void test_halfs() {
  if (!isinf(a))
    printf("FAIL at line %d\n", __LINE__ - 1);
  if (isinf(b))
    printf("FAIL at line %d\n", __LINE__ - 1);
  if (!all(isinf(va) == (short8)(-1)))
    printf("FAIL at line %d\n", __LINE__ - 1);
  if (!all(isinf(vb) == (short8)(0)))
    printf("FAIL at line %d\n", __LINE__ - 1);
}
