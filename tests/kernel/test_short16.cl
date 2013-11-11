// 32-byte vectors have different calling conventions on x86-64
// depending on whether AVX instructions are enabled or not. This is
// not encoded in the target triple. Test here whether the kernel
// library and kernels are built with compatible options.

// The failure mode is that convert_char16(short16) sets the upper 8
// vector elements to zero.

__attribute__((__noinline__))
char my_convert_char(short s)
{
  return convert_char(s);
}

__attribute__((__noinline__))
char2 my_convert_char2(short2 s2)
{
  return convert_char2(s2);
}

__attribute__((__noinline__))
char3 my_convert_char3(short3 s3)
{
  return convert_char3(s3);
}

__attribute__((__noinline__))
char4 my_convert_char4(short4 s4)
{
  return convert_char4(s4);
}

__attribute__((__noinline__))
char8 my_convert_char8(short8 s8)
{
  return convert_char8(s8);
}

__attribute__((__noinline__))
char16 my_convert_char16(short16 s16)
{
  return convert_char16(s16);
}



kernel void test_short16()
{
  short s = (short)(1);
  short2 s2 = (short2)(2);
  short3 s3 = (short3)(3);
  short4 s4 = (short4)(4);
  short8 s8 = (short8)(8);
  short16 s16 = (short16)(16);
  
  char c = my_convert_char(s);
  char2 c2 = my_convert_char2(s2);
  char3 c3 = my_convert_char3(s3);
  char4 c4 = my_convert_char4(s4);
  char8 c8 = my_convert_char8(s8);
  char16 c16 = my_convert_char16(s16);
  
  bool good = true;
  good = good && c == s;
  if (!good) {
    printf("char->short conversion failed\n");
    printf("  c=%d s=%d\n", c, s);
  }
  
  bool good2 = true;
  for (int i=0; i<2; ++i) good2 = good2 && c2[i] == s2[i];
  if (!good2) {
    printf("char2->short2 conversion failed\n");
    for (int i=0; i<2; ++i) printf("  c[%d]=%d s[%d]=%d\n", i, c2[i], i, s2[i]);
  }
  
  bool good3 = true;
  for (int i=0; i<3; ++i) good3 = good3 && c3[i] == s3[i];
  if (!good3) {
    printf("char3->short3 conversion failed\n");
    for (int i=0; i<3; ++i) printf("  c[%d]=%d s[%d]=%d\n", i, c3[i], i, s3[i]);
  }
  
  bool good4 = true;
  for (int i=0; i<4; ++i) good4 = good4 && c4[i] == s4[i];
  if (!good4) {
    printf("char4->short4 conversion failed\n");
    for (int i=0; i<4; ++i) printf("  c[%d]=%d s[%d]=%d\n", i, c4[i], i, s4[i]);
  }
  
  bool good8 = true;
  for (int i=0; i<8; ++i) good8 = good8 && c8[i] == s8[i];
  if (!good8) {
    printf("char8->short8 conversion failed\n");
    for (int i=0; i<8; ++i) printf("  c[%d]=%d s[%d]=%d\n", i, c8[i], i, s8[i]);
  }
  
  bool good16 = true;
  for (int i=0; i<16; ++i) good16 = good16 && c16[i] == s16[i];
  if (!good16) {
    printf("char16->short16 conversion failed\n");
    for (int i=0; i<16; ++i)
      printf("  c[%d]=%d s[%d]=%d\n", i, c16[i], i, s16[i]);
  }
}
