#include "common.cl"

DEFINE_BODY_V (
    test_isnan, ({
      /* isfinite */
      Jvec jres;
      jres.v = isfinite (val.v);
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          equal
              = equal
                && jres.s[n]
                       == (isfinite (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0);
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: isfinite type=%s val=%.17g res=%d good=%d\n",
                      typename, val.s[n], (int)jres.s[n],
                      (isfinite (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0));
            }
          return;
        }
      /* isinf */
      jres.v = isinf (val.v);
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          equal = equal
                  && jres.s[n]
                         == (isinf (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0);
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: isinf type=%s val=%.17g res=%d good=%d\n",
                      typename, val.s[n], (int)jres.s[n],
                      (isinf (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0));
            }
          return;
        }
      /* isnan */
      jres.v = isnan (val.v);
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          equal = equal
                  && jres.s[n]
                         == (isnan (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0);
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: isnan type=%s val=%.17g res=%d good=%d\n",
                      typename, val.s[n], (int)jres.s[n],
                      (isnan (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0));
            }
          return;
        }
      /* isnormal */
      jres.v = isnormal (val.v);
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          equal
              = equal
                && jres.s[n]
                       == (isnormal (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0);
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: isnormal type=%s val=%.17g res=%d good=%d\n",
                      typename, val.s[n], (int)jres.s[n],
                      (isnormal (val.s[n]) ? (vecsize == 1 ? +1 : -1) : 0));
            }
          return;
        }
    }))

kernel void
test_isnan ()
{
  CALL_FUNC_V (test_isnan)
}
