kernel void test_printf()
{
  printf("1\n");
  printf("%d\n", 2);
  printf("%0d\n", 3);
  printf("%.0d\n", 4);
  printf("%0.0d\n", 5);
  printf("%10d\n", 6);
  printf("%.10d\n", 7);
  printf("%10.10d\n", 8);
  printf("%v4hld\n", (int4)9);
  
  printf("1.0\n");
  printf("%f\n", 2.0f);
  printf("%0f\n", 3.0f);
  printf("%.0f\n", 4.0f);
  printf("%0.0f\n", 5.0f);
  printf("%10f\n", 6.0f);
  printf("%.10f\n", 7.0f);
  printf("%10.10f\n", 8.0f);
  printf("%v4hlf\n", (float4)9.0f);
  
  printf("|%c|%4c|%-4c|\n", 'a', 'b', 'c');
  printf("|%s|%4s|%-4s|%4s|\n", "aa", "bb", "cc", "dddddddddd");
  printf("|%p|%4p|%-4p|\n", (void*)1, (void*)2, (void*)3);
}
