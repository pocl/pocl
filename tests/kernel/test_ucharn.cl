int f2(uchar2 uc2, int z) __attribute__((noinline));
int f2(uchar2 uc2, int z)
{
int s = uc2.x + uc2.y;
return s ^ z;
}

int g2() __attribute__((noinline));
int g2()
{
uchar2 uc2 = (uchar2)(0x18, 0x29);
return f2(uc2, 0x12345678) ^ 0x12345678;
}

int f3(uchar3 uc3, int z) __attribute__((noinline));
int f3(uchar3 uc3, int z)
{
int s = uc3.x + uc3.y + uc3.z;
return s ^ z;
}

int g3() __attribute__((noinline));
int g3()
{
uchar3 uc3 = (uchar3)(0x18, 0x29, 0x3a);
return f3(uc3, 0x12345678) ^ 0x12345678;
}

int f4(uchar4 uc4, int z) __attribute__((noinline));
int f4(uchar4 uc4, int z)
{
int s = uc4.x + uc4.y + uc4.z + uc4.w;
return s ^ z;
}

int g4() __attribute__((noinline));
int g4()
{
uchar4 uc4 = (uchar4)(0x18, 0x29, 0x3a, 0x4b);
return f4(uc4, 0x12345678) ^ 0x12345678;
}

int f8(uchar8 uc8, int z) __attribute__((noinline));
int f8(uchar8 uc8, int z)
{
int s = uc8.s0 + uc8.s1 + uc8.s2 + uc8.s3 + uc8.s4 + uc8.s5 + uc8.s6 + uc8.s7;
return s ^ z;
}

int g8() __attribute__((noinline));
int g8()
{
uchar8 uc8 = (uchar8)(0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f);
return f8(uc8, 0x12345678) ^ 0x12345678;
}

int f16(uchar16 uc16, int z) __attribute__((noinline));
int f16(uchar16 uc16, int z)
{
int s = uc16.s0 + uc16.s1 + uc16.s2 + uc16.s3 + uc16.s4 + uc16.s5 + uc16.s6 + uc16.s7 + uc16.s8 + uc16.s9 + uc16.sa + uc16.sb + uc16.sc + uc16.sd + uc16.se + uc16.sf;
return s ^ z;
}

int g16() __attribute__((noinline));
int g16()
{
uchar16 uc16 = (uchar16)(0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x17, 0x28);
return f16(uc16, 0x12345678) ^ 0x12345678;
}

kernel void test_ucharn()
{
  printf("uchar2  %8x\n", f2((uchar2)(0), 0));
  printf("uchar3  %8x\n", f3((uchar3)(0), 0));
  printf("uchar4  %8x\n", f4((uchar4)(0), 0));
  printf("uchar8  %8x\n", f8((uchar8)(0), 0));
  printf("uchar16 %8x\n", f16((uchar16)(0), 0));

  printf("uchar2  %8x\n", g2());
  printf("uchar3  %8x\n", g3());
  printf("uchar4  %8x\n", g4());
  printf("uchar8  %8x\n", g8());
  printf("uchar16 %8x\n", g16());
}
