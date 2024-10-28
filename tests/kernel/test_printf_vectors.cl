kernel void test_printf_vectors()
{
  printf ("\nVECTORS\n\n");

  printf("%v4hld\n", (int4)9);
  printf("%v4hlf\n", (float4)(90.0f, 9.0f, 0.9f, 1.986546E+12));
  printf("%10.7v4hlf\n", (float4)(4096.0f, 1.0f, 0.125f, 0.0078125f));
  printf("%v4hlg\n", (float4)(90.0f, 9.0f, 0.9f, 1.986546E+33));
  printf("%v4hlF\n", (float4)(8.0f, INFINITY, -INFINITY, NAN));

  printf("%.2v4hla\n", (float4)(10.0f, 3.88E-43f, 4.0E23f, 0.0f));
  printf("%.6v4hla\n", (float4)(90.0f, 9.0f, 0.9f, 0.09f));
  printf("%.0v4hla\n", (float4)(4096.0f, 1.0f, 0.125f, 0.0078125f));

  printf("%#v2hhx\n",(char2)(0xFA,0xFB));
  printf("%#v2hx\n",(short2)(0x1234,0x8765));
  printf("%#v2hlx\n",(int2)(0x12345678,0x87654321));

  printf("\n");
  printf("uchar2   %#v2hhx\n", (uchar2)(0xa1, 0xa2));
  printf("uchar3   %#v3hhx\n", (uchar3)(0xb1, 0xb2, 0xb3));
  printf("uchar4   %#v4hhx\n", (uchar4)(0xc1, 0xc2, 0xc3, 0xc4));
  printf("uchar8   %#v8hhx\n", (uchar8)(0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8));
  printf("uchar16  %#v16hhx\n", (uchar16)(0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, 0xf1));

  printf("\n");
  printf("ushort2  %#v2hx\n", (ushort2)(0xa1a2, 0xa3a4));
  printf("ushort3  %#v3hx\n", (ushort3)(0xb1b2, 0xb3b4, 0xb5b6));
  printf("ushort4  %#v4hx\n", (ushort4)(0xc1c2, 0xc3c4, 0xc5c6, 0xc7c8));
  printf("ushort8  %#v8hx\n", (ushort8)(0xd1d2, 0xd3d4, 0xd5d6, 0xd7d8, 0xd9da, 0xdbdc, 0xddde, 0xdfe1));
  printf("ushort16 %#v16hx\n", (ushort16)(0xf1f2, 0xf3f4, 0xf5f6, 0xf7f8, 0xf9fa, 0xfbfc, 0xfdfe, 0xff11, 0x1213, 0x1415, 0x1617, 0x1819, 0x1a1b, 0x1c1d, 0x1e1f, 0x2122));

  printf("\n");
  printf("uint2    %#v2hlx\n", (uint2)(0xa1a2a3a4, 0xa5a6a7a8));
  printf("uint3    %#v3hlx\n", (uint3)(0xb1b2b3b4, 0xb5b6b7b8, 0xb9babbbc));
  printf("uint4    %#v4hlx\n", (uint4)(0xc1c2c3c4, 0xc5c6c7c8, 0xc9cacbcc, 0xcdcecfd1));
  printf("uint8    %#v8hlx\n", (uint8)(0xe1e2e3e4, 0xe5e6e7e8, 0xe9eaebec, 0xedeeeff1, 0xf2f3f4f5, 0xf6f7f8f9, 0xfafbfcfd, 0xfeff1112));
  printf("uint16   %#v16hlx\n", (uint16)(0x21222324, 0x25262728, 0x292a2b2c, 0x2d2e2f31, 0x32333435, 0x36373839, 0x3a3b3c3d, 0x3e3f4142, 0x43444546, 0x4748494a, 0x4b4c4d4f, 0x51525354, 0x55565758, 0x595a5b5c, 0x5d5e5f61, 0x62636465));

#ifdef cl_khr_int64
  printf("\n");
  printf("ulong2   %#v2lx\n", (ulong2)(0xa1a2a3a4a5a6a7a8UL, 0xa9aaabacadaeafb1UL));
  printf("ulong3   %#v3lx\n", (ulong3)(0xc1c2c3c4c5c6c7c8UL, 0xc9cacbcccdcecfd1UL, 0xd2d3d4d5d6d7d8d9UL));
  printf("ulong4   %#v4lx\n", (ulong4)(0xe1e2e3e4e5e6e7e8UL, 0xe9eaebecedeeeff1UL, 0xf2f3f4f5f6f7f8f9UL, 0xfafbfcfdfeff1112UL));
  printf("ulong8   %#v8lx\n", (ulong8)(0x2122232425262728UL, 0x292a2b2c2d2e2f31UL, 0x3233343536373839UL, 0x3a3b3c3d3e3f4142UL, 0x434445464748494aUL, 0x4b4c4d4e4f515253UL, 0x5455565758595a5bUL, 0x5c5d5e5f61626364UL));
  printf("ulong16  %#v16lx\n", (ulong16)(0x7172737475767778UL, 0x797a7b7c7d7e7f81UL, 0x8283848586878889UL, 0x8a8b8c8d8e8f9192UL, 0x939495969798999aUL, 0x9b9c9d9e9fa1a2a3UL, 0xa4a5a6a7a8a9aaabUL, 0xacadaeafb1b2b3b4UL,
                                         0xb5b6b7b8b9babbbcUL, 0xbdbebfc1c2c3c4c5UL, 0xc6c7c8c9cacbcccdUL, 0xcecfd1d2d3d4d5d6UL, 0xd7d8d9dadbdcdddeUL, 0xdfe1e2e3e4e5e6e7UL, 0xe8e9eaebecedeeefUL, 0xf1f2f3f4f5f6f7f8UL));
#endif

  printf("\n");
  printf("float2   %v2hlg\n", (float2)(1.012f, 2.022f));
  printf("float3   %v3hlg\n", (float3)(1.013f, 2.023f, 3.033f));
  printf("float4   %v4hlg\n", (float4)(1.014f, 2.024f, 3.034f, 4.044f));
  printf("float8   %v8hlg\n", (float8)(1.018f, 2.028f, 3.038f, 4.048f, 5.058f, 6.068f, 7.078f, 8.088f));
  printf("float16  %v16hlg\n", (float16)(1.01f, 2.02f, 3.03f, 4.04f, 5.05f, 6.06f, 7.07f, 8.08f, 9.09f, 10.010f, 11.011f, 12.012f, 13.013f, 14.014f, 15.015f, 16.016f));

#ifdef cl_khr_fp64
  printf("\n");
  printf("double2  %v2lg\n", (double2)(10.112, 20.222));
  printf("double3  %v3lg\n", (double3)(10.113, 20.223, 30.333));
  printf("double4  %v4lg\n", (double4)(10.114, 20.224, 30.334, 40.444));
  printf("double8  %v8lg\n", (double8)(10.118, 20.228, 30.338, 40.448, 50.558, 60.668, 70.778, 80.888));
  printf("double16 %v16lg\n", (double16)(10.11, 20.22, 30.33, 40.44, 50.55, 60.66, 70.77, 80.88, 90.99, 100.1, 110.2, 120.3, 130.4, 140.5, 150.6, 160.7));
#endif

  printf ("\nPARAMETER PASSING\n\n");

  printf("%c %#v2hhx %#v2hhx %c\n", '*', (uchar2)(0xFA, 0xFB), (char2)(0x21, 0xFD), '.');
  printf("%c %#v2hx %#v2hx %c\n", '*', (ushort2)(0x1234, 0x8765), (short2)(0xBE21, 0xF00D), '.');
  printf("%c %#v2hlx %#v2hlx %c\n", '*', (uint2)(0x12345678, 0x87654321), (int2)(0x2468ACE0, 0xFDB97531), '.');
  printf("%c %#v2hhx %#v2hhx %#v2hhx %#v2hhx %#v2hhx %#v2hhx %#v2hhx %#v2hhx %c\n",
         '*',
         (uchar2)(0xFA,0xFB),
         (uchar2)(0xFC,0xFD),
         (uchar2)(-23,-42),
         (uchar2)(0xFE,0xFF),
         (uchar2)(0x21,0x2B),
         (uchar2)(0x3A,0x3B),
         (uchar2)(0x4A,0x4B),
         (uchar2)(0x5A,0x5B),
         '.');

  printf("\n%c %#v2hhx %#v2hhx %c\n", 'c',
         (uchar2)(0xa1, 0xa2),
         (uchar2)(0x21, 0xb4), '.');
  printf("%c %#v3hhx %#v3hhx %c\n", 'c',
         (uchar3)(0xc1, 0xc2, 0x21),
         (uchar3)(0xd4, 0xd5, 0xd6), '.');
  printf("%c %#v4hhx %#v4hhx %c\n", 'c',
         (uchar4)(0xe1, 0xe2, 0x21, 0xe4),
         (uchar4)(0xf5, 0xf6, 0xf7, 0xf8), '.');
  printf("%c %#v8hhx %#v8hhx %c\n", 'c',
         (uchar8)(0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18),
         (uchar8)(0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x31), '.');
  printf("%c %#v16hhx %#v16hhx %c\n", 'c',
         (uchar16)(0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x21, 0x4d, 0x4e, 0x4f, 0x51),
         (uchar16)(0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x61, 0x62), '.');
  printf("%c %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %#hhx %c\n", 'c',
         0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x81,
         0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x91, 0x92, '.');

  printf("\n%c %#v2hx %#v2hx %c\n", 's',
         (ushort2)(0xa1a2, 0xa3a4),
         (ushort2)(0xb521, 0xb7b8), '.');
  printf("%c %#v3hx %#v3hx %c\n", 's',
         (ushort3)(0xc1c2, 0xc3c4, 0xc5c6),
         (ushort3)(0xd7d8, 0xd921, 0xdbdc), '.');
  printf("%c %#v4hx %#v4hx %c\n", 's',
         (ushort4)(0xe1e2, 0xe3e4, 0xe5e6, 0xe7e8),
         (ushort4)(0xf9fa, 0xfbfc, 0xfdfe, 0xff11), '.');
  printf("%c %#v8hx %#v8hx %c\n", 's',
         (ushort8)(0x2122, 0x2324, 0x2526, 0x2728, 0x292a, 0x2b2c, 0x2d2e, 0x2f31),
         (ushort8)(0x3233, 0x3435, 0x3637, 0x3821, 0x3a3b, 0x3c3d, 0x3e3f, 0x4142), '.');
  printf("%c %#v16hx %#v16hx %c\n", 's',
         (ushort16)(0x5152, 0x5354, 0x5556, 0x5758, 0x595a, 0x5b5c, 0x5d5e, 0x5f61, 0x6263, 0x6465, 0x6667, 0x6869, 0x6a6b, 0x6c6d, 0x6e6f, 0x7172),
         (ushort16)(0x7374, 0x7576, 0x7778, 0x797a, 0x7b7c, 0x7d7e, 0x7f81, 0x8221, 0x8485, 0x8687, 0x8889, 0x8a8b, 0x8c8d, 0x8e8f, 0x9192, 0x9394), '.');
  printf("%c %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %#hx %c\n", 's',
         0xa1a2, 0xa3a4, 0xa5a6, 0xa7a8, 0xa9aa, 0xabac, 0xadae, 0xafb1, 0xb2b3, 0xb4b5, 0xb6b7, 0xb8b9, 0xbabb, 0xbcbd, 0xbebf, 0xc1c2,
         0xc3c4, 0xc5c6, 0xc7c8, 0xc9ca, 0xcbcc, 0xcdce, 0xcfd1, 0xd2d3, 0xd4d5, 0xd6d7, 0xd8d9, 0xdadb, 0xdcdd, 0xdedf, 0xe1e2, 0xe3e4, '.');

  printf("\n%c %#v2hlx %#v2hlx %c\n", 'i',
         (uint2)(0xa1a2a3a4, 0xa5a6a7a8),
         (uint2)(0xb9babbbc, 0xbdbebfc1), '.');
  printf("%c %#v3hlx %#v3hlx %c\n", 'i',
         (uint3)(0xd1d2d3d4, 0xd5d6d7d8, 0xd9dadbdc),
         (uint3)(0xedeeeff1, 0xf2f3f4f5, 0xf6f7f8f9), '.');
  printf("%c %#v4hlx %#v4hlx %c\n", 'i',
         (uint4)(0x11121314, 0x15161718, 0x191a1b1c, 0x1d1e1f21),
         (uint4)(0x22232425, 0x26272829, 0x2a2b2c2d, 0x2e2f3132), '.');
  printf("%c %#v8hlx %#v8hlx %c\n", 'i',
         (uint8)(0x41424344, 0x45464748, 0x494a4b4c, 0x4d4e4f51, 0x51535455, 0x35575859, 0x5a5b5c5d, 0x5e5f6162),
         (uint8)(0x63646566, 0x6768696a, 0x6b6c6d6e, 0x6f717273, 0x74757677, 0x78797a7b, 0x7c7d7e7f, 0x81828384), '.');
  printf("%c %#v16hlx %#v16hlx %c\n", 'i',
         (uint16)(0x91929394, 0x95969798, 0x999a9b9c, 0x9d9e9fa1, 0xa2a3a4a5, 0xa6a7a8a9, 0xaaabacad, 0xaeafb1b2, 0xb3b4b5b6, 0xb7b8b9ba, 0xbbbcbdbe, 0xbfc1c2c3, 0xc4c5c6c7, 0xc8c9cacb, 0xcccdcecf, 0xd1d2d3d4),
         (uint16)(0xd5d6d7d8, 0xd9dadbdc, 0xdddedfe1, 0xe2e3e4e5, 0xe6e7e8e9, 0xeaebeced, 0xeeeff1f2, 0xf3f4f5f6, 0xf7f8f9fa, 0xfbfcfdfe, 0xff111213, 0x14151617, 0x18191a1b, 0x1c1d1e1f, 0x21222324, 0x25262728), '.');
  printf("%c %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %#x %c\n", 'i',
         0x31323334, 0x35363738, 0x393a3b3c, 0x3d3e3f41, 0x42434445, 0x46474849, 0x4a4b4c4d, 0x4e4f5152, 0x53545556, 0x5758595a, 0x5b5c5d5e, 0x5f616263, 0x64656667, 0x68696a6b, 0x6c6d6e6f, 0x71727374,
         0x75767778, 0x797a7b7c, 0x7d7e7f81, 0x82838485, 0x86878889, 0x8a8b8c8d, 0x8e8f9192, 0x93949596, 0x9798999a, 0x9b9c9d9e, 0x9fa1a2a3, 0xa4a5a6a7, 0xa8a9aaab, 0xacadaeaf, 0xb1b2b3b4, 0xb5b6b7b8, '.');

#ifdef cl_khr_int64
  printf("\n%c %#v2lx %#v2lx %c\n", 'l',
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
  printf("%c %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %#lx %c\n", 'l',
         0x4142434445464748UL, 0x494a4b4c4d4e4f51UL, 0x5253545556575859UL, 0x5a5b5c5d5e5f6162UL, 0x636465666768696aUL, 0x6b6c6d6e6f717273UL, 0x7475767778797a7bUL, 0x7c7d7e7f81828384UL,
         0x85868788898a8b8cUL, 0x8d8e8f9192939495UL, 0x969798999a9b9c9dUL, 0x9e9fa1a2a3a4a5a6UL, 0xa7a8a9aaabacadaeUL, 0xafb1b2b3b4b5b6b7UL, 0xb8b9babbbcbdbebfUL, 0xc1c2c3c4c5c6c7c8UL,
         0xc9cacbcccdcecfd1UL, 0xd2d3d4d5d6d7d8d9UL, 0xdadbdcdddedfe1e2UL, 0xe3e4e5e6e7e8e9eaUL, 0xebecedeeeff1f2f3UL, 0xf4f5f6f7f8f9fafbUL, 0xfcfdfeff11121314UL, 0x15161718191a1b1cUL,
         0x1d1e1f2122232425UL, 0x262728292a2b2c2dUL, 0x2e2f313233343536UL, 0x3738393a3b3c3d3eUL, 0x3f41424344454647UL, 0x48494a4b4c4d4e4fUL, 0x5152535455565758UL, 0x595a5b5c5d5e5f61UL, '.');
#endif

  printf("\n%c %v2hlg %v2hlg %c\n", 'f',
         (float2)(21.1f, 21.2f),
         (float2)(22.3f, 22.4f), '.');
  printf("%c %v3hlg %v3hlg %c\n", 'f',
         (float3)(31.1f, 31.2f, 31.3f),
         (float3)(32.4f, 32.5f, 32.6f), '.');
  printf("%c %v4hlg %v4hlg %c\n", 'f',
         (float4)(41.1f, 41.2f, 41.3f, 41.4f),
         (float4)(42.5f, 42.6f, 42.7f, 42.8f), '.');
  printf("%c %v8hlg %v8hlg %c\n", 'f',
         (float8)(81.01f, 81.02f, 81.03f, 81.04f, 81.05f, 81.06f, 81.07f, 81.08f),
         (float8)(82.09f, 82.10f, 82.11f, 82.12f, 82.13f, 82.14f, 82.15f, 82.16f), '.');
  printf("%c %v16hlg %v16hlg %c\n", 'f',
         (float16)(1.01f, 1.02f, 1.03f, 1.04f, 1.05f, 1.06f, 1.07f, 1.08f, 1.09f, 1.10f, 1.11f, 1.12f, 1.13f, 1.14f, 1.15f, 1.16f),
         (float16)(2.17f, 2.18f, 2.19f, 2.20f, 2.21f, 2.22f, 2.23f, 2.24f, 2.25f, 2.26f, 2.27f, 2.28f, 2.29f, 2.30f, 2.31f, 2.32f), '.');
  printf("%c %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %c\n", 'f',
         1.4f, 2.4f, 3.4f, 4.4f, 5.4f, 6.4f, 7.4f, 8.4f, 9.4f, 10.4f, 11.4f, 12.4f, 13.4f, 14.4f, 15.4f, 16.4f,
         17.4f, 18.4f, 19.4f, 20.4f, 21.4f, 22.4f, 23.4f, 24.4f, 25.4f, 26.4f, 27.4f, 28.4f, 29.4f, 30.4f, 31.4f, 32.4f, '.');

#ifdef cl_khr_fp64
  printf("\n%c %v2lg %v2lg %c\n", 'd',
         (double2)(21.1, 21.2),
         (double2)(22.3, 22.4), '.');
  printf("%c %v3lg %v3lg %c\n", 'd',
         (double3)(31.1, 31.2, 31.3),
         (double3)(32.4, 32.5, 32.6), '.');
  printf("%c %v4lg %v4lg %c\n", 'd',
         (double4)(41.1, 41.2, 41.3, 41.4),
         (double4)(42.5, 42.6, 42.7, 42.8), '.');
  printf("%c %v8lg %v8lg %c\n", 'd',
         (double8)(81.01, 81.02, 81.03, 81.04, 81.05, 81.06, 81.07, 81.08),
         (double8)(82.09, 82.10, 82.11, 82.12, 82.13, 82.14, 82.15, 82.16), '.');
  printf("%c %v16lg %v16lg %c\n", 'd',
         (double16)(1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16),
         (double16)(2.17, 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.30, 2.31, 2.32), '.');
  printf("%c %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %c\n", 'd',
         1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 16.8,
         17.8, 18.8, 19.8, 20.8, 21.8, 22.8, 23.8, 24.8, 25.8, 26.8, 27.8, 28.8, 29.8, 30.8, 31.8, 32.8, '.');
#endif

  printf("\n%c %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %c\n", 'x',
         0x31323334, 2.4f, 0x393a3b3c, 4.4f, 0x42434445, 6.4f, 0x4a4b4c4d, 8.4f, 0x53545556, 10.4f, 0x5b5c5d5e, 12.4f, 0x64656667, 14.4f, 0x6c6d6e6f, 16.4f,
         0x75767778, 18.4f, 0x7d7e7f81, 20.4f, 0x86878889, 22.4f, 0x8e8f9192, 24.4f, 0x9798999a, 26.4f, 0x9fa1a2a3, 28.4f, 0xa8a9aaab, 30.4f, 0xb1b2b3b4, 32.4f, '.');
  printf("%c %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %g %#x %c\n", 'x',
         1.4f, 0x35363738, 3.4f, 0x3d3e3f41, 5.4f, 0x46474849, 7.4f, 0x4e4f5152, 9.4f, 0x5758595a, 11.4f, 0x5f616263, 13.4f, 0x68696a6b, 15.4f, 0x71727374,
         17.4f, 0x797a7b7c, 19.4f, 0x82838485, 21.4f, 0x8a8b8c8d, 23.4f, 0x93949596, 25.4f, 0x9b9c9d9e, 27.4f, 0xa4a5a6a7, 29.4f, 0xacadaeaf, 31.4f, 0xb5b6b7b8, '.');
}
