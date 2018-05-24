#include "common.cl"

DEFINE_BODY_V (
    test_fabs, ({
      /* fabs */
      res.v = fabs (val.v);
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          S r, g;
          r.s = res.s[n];
          g.s = good.s[n];
          equal = equal && (ISNAN (val.s[n]) || r.sj == g.sj);
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: fabs type=%s val=%.17g res=%.17g good=%.17g\n",
                      typename, val.s[n], res.s[n], good.s[n]);
            }
          return;
        }
    }))

kernel void test_fabs()
{
  CALL_FUNC_V(test_fabs)
}
