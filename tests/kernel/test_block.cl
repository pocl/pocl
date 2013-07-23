void output(float (^op)(float))
{
  printf("op(1)=%f\n", op(1.0f));
}

kernel void test_block()
{
  // Two simple tests
  float (^add1)(float) = ^(float x) { return x + 1.0f; };
  float (^mul2)(float) = ^(float x) { return x * 2.0f; };
  output(add1);
  output(mul2);
  
  // Capture a pointer to a local variable
  float counter = 0.0f;
  float* counterp = &counter;
  float (^stepcounter)(float) = ^(float x) {
    return *counterp += x;
  };
  output(stepcounter);
  output(stepcounter);
}
