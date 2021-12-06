ulong f2(ulong2 ul2, ulong z) __attribute__((noinline));
ulong f2(ulong2 ul2, ulong z)
{
ulong s = ul2.x + ul2.y;
return s ^ z;
}

ulong g2() __attribute__((noinline));
ulong g2()
{
ulong2 ul2 = (ulong2)(0x18, 0x29);
return -f2(ul2, 0x123456789abcdef1UL);
}

ulong f3(ulong3 ul3, ulong z) __attribute__((noinline));
ulong f3(ulong3 ul3, ulong z)
{
ulong s = ul3.x + ul3.y + ul3.z;
return s ^ z;
}

ulong g3() __attribute__((noinline));
ulong g3()
{
ulong3 ul3 = (ulong3)(0x18, 0x29, 0x3a);
return -f3(ul3, 0x123456789abcdef1UL);
}

ulong f4(ulong4 ul4, ulong z) __attribute__((noinline));
ulong f4(ulong4 ul4, ulong z)
{
ulong s = ul4.x + ul4.y + ul4.z + ul4.w;
return s ^ z;
}

ulong g4() __attribute__((noinline));
ulong g4()
{
ulong4 ul4 = (ulong4)(0x18, 0x29, 0x3a, 0x4b);
return -f4(ul4, 0x123456789abcdef1UL);
}

ulong f8(ulong8 ul8, ulong z) __attribute__((noinline));
ulong f8(ulong8 ul8, ulong z)
{
ulong s = ul8.s0 + ul8.s1 + ul8.s2 + ul8.s3 + ul8.s4 + ul8.s5 + ul8.s6 + ul8.s7;
return s ^ z;
}

ulong g8() __attribute__((noinline));
ulong g8()
{
ulong8 ul8 = (ulong8)(0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f);
return -f8(ul8, 0x123456789abcdef1UL);
}

ulong f16(ulong16 ul16, ulong z) __attribute__((noinline));
ulong f16(ulong16 ul16, ulong z)
{
ulong s = ul16.s0 + ul16.s1 + ul16.s2 + ul16.s3 + ul16.s4 + ul16.s5 + ul16.s6 + ul16.s7 + ul16.s8 + ul16.s9 + ul16.sa + ul16.sb + ul16.sc + ul16.sd + ul16.se + ul16.sf;
return s ^ z;
}

ulong g16() __attribute__((noinline));
ulong g16()
{
ulong16 ul16 = (ulong16)(0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x17, 0x28);
return -f16(ul16, 0x123456789abcdef1UL);
}

int pl2() __attribute__((noinline));
int pl2()
{
  printf("ulong2   %#v2lx\n", (ulong2)(0xa1a2a3a4a5a6a7a8UL, 0xa9aaabacadaeafb1UL));
  return 0;
}

int pl3() __attribute__((noinline));
int pl3()
{
  printf("ulong3   %#v3lx\n", (ulong3)(0xc1c2c3c4c5c6c7c8UL, 0xc9cacbcccdcecfd1UL, 0xd2d3d4d5d6d7d8d9UL));
  return 0;
}

int pl4() __attribute__((noinline));
int pl4()
{
  printf("ulong4   %#v4lx\n", (ulong4)(0xe1e2e3e4e5e6e7e8UL, 0xe9eaebecedeeeff1UL, 0xf2f3f4f5f6f7f8f9UL, 0xfafbfcfdfeff1112UL));
  return 0;
}

int pl8() __attribute__((noinline));
int pl8()
{
  printf("ulong8   %#v8lx\n", (ulong8)(0x2122232425262728UL, 0x292a2b2c2d2e2f31UL, 0x3233343536373839UL, 0x3a3b3c3d3e3f4142UL, 0x434445464748494aUL, 0x4b4c4d4e4f515253UL, 0x5455565758595a5bUL, 0x5c5d5e5f61626364UL));
  return 0;
}

int pl16() __attribute__((noinline));
int pl16()
{
  printf("ulong16  %#v16lx\n", (ulong16)(0x7172737475767778UL, 0x797a7b7c7d7e7f81UL, 0x8283848586878889UL, 0x8a8b8c8d8e8f9192UL, 0x939495969798999aUL, 0x9b9c9d9e9fa1a2a3UL, 0xa4a5a6a7a8a9aaabUL, 0xacadaeafb1b2b3b4UL,
                                         0xb5b6b7b8b9babbbcUL, 0xbdbebfc1c2c3c4c5UL, 0xc6c7c8c9cacbcccdUL, 0xcecfd1d2d3d4d5d6UL, 0xd7d8d9dadbdcdddeUL, 0xdfe1e2e3e4e5e6e7UL, 0xe8e9eaebecedeeefUL, 0xf1f2f3f4f5f6f7f8UL));
  return 0;
}

kernel void test_printf_vectors_ulongn()
{
  printf("ulong2   %#v2lx\n", (ulong2)(0xa1a2a3a4a5a6a7a8UL, 0xa9aaabacadaeafb1UL));
  printf("ulong3   %#v3lx\n", (ulong3)(0xc1c2c3c4c5c6c7c8UL, 0xc9cacbcccdcecfd1UL, 0xd2d3d4d5d6d7d8d9UL));
  printf("ulong4   %#v4lx\n", (ulong4)(0xe1e2e3e4e5e6e7e8UL, 0xe9eaebecedeeeff1UL, 0xf2f3f4f5f6f7f8f9UL, 0xfafbfcfdfeff1112UL));
  printf("ulong8   %#v8lx\n", (ulong8)(0x2122232425262728UL, 0x292a2b2c2d2e2f31UL, 0x3233343536373839UL, 0x3a3b3c3d3e3f4142UL, 0x434445464748494aUL, 0x4b4c4d4e4f515253UL, 0x5455565758595a5bUL, 0x5c5d5e5f61626364UL));
  printf("ulong16  %#v16lx\n", (ulong16)(0x7172737475767778UL, 0x797a7b7c7d7e7f81UL, 0x8283848586878889UL, 0x8a8b8c8d8e8f9192UL, 0x939495969798999aUL, 0x9b9c9d9e9fa1a2a3UL, 0xa4a5a6a7a8a9aaabUL, 0xacadaeafb1b2b3b4UL,
                                         0xb5b6b7b8b9babbbcUL, 0xbdbebfc1c2c3c4c5UL, 0xc6c7c8c9cacbcccdUL, 0xcecfd1d2d3d4d5d6UL, 0xd7d8d9dadbdcdddeUL, 0xdfe1e2e3e4e5e6e7UL, 0xe8e9eaebecedeeefUL, 0xf1f2f3f4f5f6f7f8UL));

  pl2();
  pl3();
  pl4();
  pl8();
  pl16();

  printf("\n");
  printf("%c %#v2lx %#v2lx %c\n", 'l',
         (ulong2)(0xa1a2a3a4a5a6a7a8UL, 0xa9aaabacadaeafb1UL),
         (ulong2)(0xb2b3b4b5b6b7b8b9UL, 0xbabbbcbdbebfc1c2UL), '.');
  printf("%c %#v3lx %#v3lx %c\n", 'l',
         (ulong3)(0xd1d2d3d4d5d6d7d8UL, 0xd9dadbdcdddedfe1UL, 0xe2e3e4e5e6e7e8e9UL),
         (ulong3)(0xeaebecedeeeff1f2UL, 0xf3f4f5f6f7f8f9faUL, 0xfbfcfdfeff111213UL), '.');
  printf("%c %#v4lx %#v4lx %c\n", 'l',
         (ulong4)(0x2122232425262728UL, 0x292a2b2c2d2e2f31UL, 0x3233343536373839UL, 0x3a3b3c3d3e3f4142UL),
         (ulong4)(0x434445464748494aUL, 0x4b4c4d4e4f515253UL, 0x5455565758595a5bUL, 0x5c5d5e5f61626364UL), '.');
  printf("%c %#v8lx %#v8lx %c\n", 'l',
         (ulong8)(0x7172737475767778UL, 0x797a7b7c7d7e7f81UL, 0x8283848586878889UL, 0x8a8b8c8d8e8f9192UL, 0x939495969798999aUL, 0x9b9c9d9e9fa1a2a3UL, 0xa4a5a6a7a8a9aaabUL, 0xacadaeafb1b2b3b4UL),
         (ulong8)(0xb5b6b7b8b9babbbcUL, 0xbdbebfc1c2c3c4c5UL, 0xc6c7c8c9cacbcccdUL, 0xcecfd1d2d3d4d5d6UL, 0xd7d8d9dadbdcdddeUL, 0xdfe1e2e3e4e5e6e7UL, 0xe8e9eaebecedeeefUL, 0xf1f2f3f4f5f6f7f8UL), '.');
  printf("%c %#v16lx %#v16lx %c\n", 'l',
         (ulong16)(0x1112131415161718UL, 0x191a1b1c1d1e1f21UL, 0x2223242526272829UL, 0x2a2b2c2d2e2f3132UL, 0x333435363738393aUL, 0x3b3c3d3e3f414243UL, 0x4445464748494a4bUL, 0x4c4d4e4f51525354UL,
                   0x55565758595a5b5cUL, 0x5d5e5f6162636465UL, 0x666768696a6b6c6dUL, 0x6e6f717273747576UL, 0x7778797a7b7c7d7eUL, 0x7f81828384858687UL, 0x88898a8b8c8d8e8fUL, 0x9192939495969798UL),
         (ulong16)(0x999a9b9c9d9e9fa1UL, 0xa2a3a4a5a6a7a8a9UL, 0xaaabacadaeafb1b2UL, 0xb3b4b5b6b7b8b9baUL, 0xbbbcbdbebfc1c2c3UL, 0xc4c5c6c7c8c9cacbUL, 0xcccdcecfd1d2d3d4UL, 0xd5d6d7d8d9dadbdcUL,
                   0xdddedfe1e2e3e4e5UL, 0xe6e7e8e9eaebecedUL, 0xeeeff1f2f3f4f5f6UL, 0xf7f8f9fafbfcfdfeUL, 0xff11121314151617UL, 0x18191a1b1c1d1e1fUL, 0x2122232425262728UL, 0x292a2b2c2d2e2f31UL), '.');

  printf("\n");
  printf("ulong2  %16lx\n", f2((ulong2)(0), 0));
  printf("ulong3  %16lx\n", f3((ulong3)(0), 0));
  printf("ulong4  %16lx\n", f4((ulong4)(0), 0));
  printf("ulong8  %16lx\n", f8((ulong8)(0), 0));
  printf("ulong16 %16lx\n", f16((ulong16)(0), 0));

  printf("ulong2  %16lx\n", g2());
  printf("ulong3  %16lx\n", g3());
  printf("ulong4  %16lx\n", g4());
  printf("ulong8  %16lx\n", g8());
  printf("ulong16 %16lx\n", g16());
}
