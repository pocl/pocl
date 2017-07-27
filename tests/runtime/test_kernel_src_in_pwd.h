#ifndef __FUNC__
#error __FUNC__ macro did not propagate to the Program
#endif

#define STR(a) S (a)
#define S(a) #a

void
__FUNC__ ()
{
  printf ("inside __FUNC__ (%s)\n", STR (__FUNC__));
}

kernel
void test_kernel() {
    __FUNC__();
}
