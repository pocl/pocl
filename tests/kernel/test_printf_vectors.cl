kernel void test_printf_vectors()
{
  printf ("\nVECTORS\n\n");

  printf("%v4hld\n", (int4)9);
  printf("%v4hlf\n", (float4)(90.0f, 9.0f, 0.9f, 1.986546E+12));
  printf("%10.7v4hlf\n", (float4)(4096.0f, 1.0f, 0.125f, 0.0078125f));
  printf("%v4hlg\n", (float4)(90.0f, 9.0f, 0.9f, 1.986546E+33));
  printf("%v4hlF\n", (float4)(8.0f, INFINITY, -INFINITY, NAN));

  printf("%v4hla\n", (float4)(10.0f, 3.88E-43f, 4.0E23f, 0.0f));
  printf("%v4hla\n", (float4)(90.0f, 9.0f, 0.9f, 0.09f));
  printf("%v4hla\n", (float4)(4096.0f, 1.0f, 0.125f, 0.0078125f));

  printf("%#v2hhx\n",(char2)(0xFA,0xFB));
  printf("%#v2hx\n",(short2)(0x1234,0x8765));
  printf("%#v2hlx\n",(int2)(0x12345678,0x87654321));

  printf ("\nPARAMETER PASSING\n\n");

  printf("%c %#v2hhx %#v2hhx %c\n", '*', (char2)(0xFA, 0xFB), (char2)(0xFC, 0xFD), '.');
  printf("%c %#v2hx %#v2hx %c\n", '*', (short2)(0x1234, 0x8765), (short2)(0xBEEF, 0xF00D), '.');
  printf("%c %#v2hlx %#v2hlx %c\n", '*', (int2)(0x12345678, 0x87654321), (int2)(0x2468ACE0, 0xFDB97531), '.');
}
