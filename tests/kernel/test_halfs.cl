#pragma OPENCL EXTENSION cl_khr_fp16 : enable

volatile global half a = INFINITY;
volatile global half b = 1.0h;
volatile global half8 va = (half8)(INFINITY);
volatile global half8 vb = (half8)(1.0h);

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
