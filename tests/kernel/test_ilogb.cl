#include "common.cl"

DEFINE_BODY_V (
    test_ilogb, ({
      /* ilogb */
      Ivec ires;
      ires.v = ilogb (val.v);
      Ivec igoodres;
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          if (ISNAN (val.s[n]))
            {
              igoodres.s[n] = FP_ILOGBNAN;
            }
          else if (val.s[n] == (stype)0)
            {
              igoodres.s[n] = FP_ILOGB0;
            }
          else if (isinf (val.s[n]))
            {
              igoodres.s[n] = INT_MAX;
            }
          else
            {
              // We round down to "correct" for inaccuracies in log2
              // We divide by 2 since log2 is wrong for large inputs
              igoodres.s[n]
                  = 1
                    + rint (floor (0.999999f
                                   * log2 (0.5 * fabs (val.s[n]))));
            }
          equal = equal && ires.s[n] == igoodres.s[n];
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: ilogb type=%s val=%.17g res=%d good=%d\n",
                      typename, val.s[n], (int)ires.s[n], (int)igoodres.s[n]);
            }
          return;
        }
    }))


kernel void
test_ilogb ()
{
  CALL_FUNC_V (test_ilogb)
}
