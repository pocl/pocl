kernel void test_printf()
{

  printf("");
  printf("INTEGERS\n\n");

  printf("1\n");
  printf("%d\n", 2);
  printf("%0d\n", 3);
  printf("%.0d\n", 4);
  printf("%0.0d\n", 5);
  printf("%10d\n", 6);
  printf("%.10d\n", 7);
  printf("%10.10d\n", 8);
  printf("%5.10d\n", 9);
  printf("%9.4d\n", 10);
  printf("%-06i\n", 10);

  printf("%i\n", INT_MIN);
  printf("%li\n", LONG_MIN);

  printf("%u\n", INT_MAX);
  printf("%lu\n", LONG_MAX);

  printf("%#o\n",100000000);
  printf("%o\n",100000000);
  printf("%#o\n",0);
  printf("%o\n",0);

  printf("%.0o\n",0);
  printf("%.0i\n",0);

  printf("%4c\n",'1');
  printf("%-4c\n",'1');
  printf("%c\n",66);

  printf("%.0u\n",0);
  printf("%#X\n",0);

  printf("%s\n", (void*)0);

  printf ("\nFLOATS\n");

  printf ("\n%%f conversion\n\n");

  printf("1.0\n");
  printf("%f\n", 2.0f);
  printf("%0f\n", 3.0f);
  printf("%.0f\n", 4.0f);
  printf("%0.0f\n", 5.0f);
  printf("%10f\n", 6.0f);
  printf("%.10f\n", 7.0f);
  printf("%10.10f\n\n\n", 8.0f);

  printf("%f\n", 0.0078125f);
  printf("%f\n",10.3456);
  printf("%.1f\n",10.3456);
  printf("%.2f\n",10.3456);
  printf("%.3f\n",0.0356);
  printf("%8.3f\n",10.3456);
  printf("%08.2f\n",10.3456);
  printf("%-8.2f\n",10.3456);
  printf("%+8.2f\n",-10.3456);

  printf("%.0f\n",0.0f);
  printf("%.0f\n",0.1f);
  printf("%.0f\n",0.6f);
  printf("%.2f\n",0.125f);

  printf("%f\n", 0.0f);
  printf("%012f\n", 0.0f);
  printf("%0.3f\n", 0.0f);

  printf("%+8.2f\n",-10.3456);

  printf("%f\n", NAN);

  printf ("\n%%e conversion\n\n");

  printf("%e\n", 0.0f);
  printf("%014e\n", 0.0f);
  printf("%0.3e\n", 0.0f);

  printf("%.2e\n",10.3456f);
  printf("%.3e\n",10.3456f);
  printf("%.4e\n",10.3456f);

  /* test RTE rounding */
  printf("%.6e %.8e \n ", -252569.750, -252569.750);
  printf("%.6e %.8e \n ", 4184049.50, 4184049.50);
  printf("%.1e\n",  1.25E+15);

  float j = as_float((uint)0x408fffffU);
  float k = as_float((uint)0x40f00e00U);

  printf ("\n%%a conversion\n\n");

  printf("%a\n", 0.0f);
  printf("%012a\n", 0.0f);
  printf("%0.3a\n", 0.0f);

  printf("%16.5A\n", j);
  printf("%10.1a\n", j);

  printf("%4.0a\n", k);
  printf("%4.1a\n", k);
  printf("%a\n", k);
  printf("%a\n", 4.0f);
  printf("%a\n", 0.0f);
  printf("%014.2a\n", k);
  printf("%10a\n", 10.0f);
  printf("%.6a\n",0.1);

  printf ("\nMODIFIERS\n\n");

  printf ("%4i\n",0);
  printf ("%04i\n",0);
  printf ("%+4i\n",0);
  printf ("% 04i\n",0);
  printf ("%+04i\n",0);
  printf ("%+-4i\n",0);
  printf ("%-4i\n",0);
  printf ("% -4i\n",0);

  printf ("%4i\n",34);
  printf ("%04i\n",34);
  printf ("%+4i\n",34);
  printf ("% 04i\n",34);
  printf ("%+04i\n",34);
  printf ("%+-4i\n",34);
  printf ("%-4i\n",34);
  printf ("% -4i\n",34);

  printf ("%4.1f\n", M_PI);
  printf ("%-4.1f\n", M_PI);
  printf ("%+4.1f\n", M_PI);
  printf ("%+-4.1f\n", M_PI);
  printf ("%04.1f\n", M_PI);
  printf ("%+04.1f\n", M_PI);
  printf ("% 04.1f\n", M_PI);
  printf ("%- 4.1f\n", M_PI);

  printf ("%4s\n","");
  printf ("%-4s\n","");
  printf ("%4s\n","je");
  printf ("%-4s\n","je");
  printf ("%4s\n","quickfoxjump");
  printf ("%-4s\n","quickfoxjump");

  printf ("%.0f\n", M_PI);
  printf ("%.1f\n", M_PI);
  printf ("%.2f\n", M_PI);
  printf ("%.3f\n", M_PI);
  printf ("%.4f\n", M_PI);
  printf ("%.5f\n", M_PI);
  printf ("%.6f\n", M_PI);
  printf ("%.7f\n", M_PI);

  printf ("%4.0f\n", M_PI);
  printf ("%4.2f\n", M_PI);
  printf ("%4.6f\n", M_PI);
  printf ("%4.7f\n", M_PI);

  printf("|%c|%4c|%-4c|\n", 'a', 'b', 'c');
  printf("|%s|%4s|%-4s|%4s|%.4s|\n", "aa", "bb", "cc", "dddddddddd", "eeeeee");
  printf("|%p|%12p|%-12p|\n", (void*)0x2349aacc, (void*)0xdeaddeed, (void*)0x92820384);
}
