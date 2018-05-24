#include "common.cl"

DEFINE_BODY_V (
    test_copy_signbit, ({
      /* signbit */
      Jvec jres;
      jres.v = signbit (val.v);
      equal = true;
      for (int n = 0; n < vecsize; ++n)
        {
          equal = equal
                  && (ISNAN (val.s[n])
                      || jres.s[n] == (sign > 0 ? 0 : vecsize == 1 ? +1 : -1));
        }
      if (!equal)
        {
          for (int n = 0; n < vecsize; ++n)
            {
              printf ("FAIL: signbit type=%s val=%.17g res=%d good=%d\n",
                      typename, val.s[n], (int)jres.s[n],
                      (sign > 0 ? 0 : vecsize == 1 ? +1 : -1));
            }
          return;
        }

      /* copysign */
      for (int sign2 = -1; sign2 <= +1; sign2 += 2)
        {
          res.v = copysign (val.v, (stype)sign2 * val2.v);
          equal = true;
          for (int n = 0; n < vecsize; ++n)
            {
              S r, g;
              r.s = res.s[n];
              g.s = sign2 * good.s[n];
              equal
                  = equal
                    && (ISNAN (val.s[n]) || ISNAN (val2.s[n]) || r.sj == g.sj);
            }
          if (!equal)
            {
              for (int n = 0; n < vecsize; ++n)
                {
                  printf ("FAIL: copysign type=%s val=%.17g sign=%.17g "
                          "res=%.17g good=%.17g\n",
                          typename, val.s[n], sign2 * val2.s[n], res.s[n],
                          good.s[n]);
                }
              return;
            }
        }
    }))

kernel void
test_copy_signbit ()
{
  CALL_FUNC_V (test_copy_signbit)
}
