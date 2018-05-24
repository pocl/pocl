#include "common.cl"

DEFINE_BODY_V (
    test_ldexp, ({
      /* ldexp */
      Ivec ival2;
      ival2.v = (itype)4;
      res.v = ldexp (val.v, ival2.v);
      Tvec goodres;
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          if (ISNAN (val.s[n]))
            {
              goodres.s[n] = NAN;
            }
          else if (val.s[n] == (stype)0)
            {
              goodres.s[n] = val.s[n];
            }
          else if (isinf (val.s[n]))
            {
              goodres.s[n] = val.s[n];
            }
          else
            {
              goodres.s[n] = val.s[n] * (stype) (16.0);
            }
          equal = equal && ISEQ (res.s[n], goodres.s[n]);
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: ldexp type=%s val=%.17g val2=%d res=%.17g "
                      "good=%.17g\n",
                      typename, val.s[n], (int)ival2.s[n], res.s[n],
                      goodres.s[n]);
            }
          return;
        }
    }))


kernel void
test_ldexp ()
{
  CALL_FUNC_V (test_ldexp)
}
