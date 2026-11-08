#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_abs, ({
                      // abs
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal && res_abs.s[n] == good_abs.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: abs type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)good_abs.s[n],
                                      (int)res_abs.s[n]);
                            }
                          error = true;
                        }
                      // abs of the minimum value is 2^(bits-1), so the top
                      // bit of the result is set. Computing it as a signed
                      // negation overflows, after which LLVM may assume that
                      // the bit is clear.
                      Tvec m, res_top;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          m.s[n] = x.s[n] & tmin;
                        }
                      res_top.u = abs (m.v) >> (sugtype)(bits - 1);
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal && res_top.s[n] == (m.s[n] != 0);
                        }
                      if (!equal)
                        {
                          printf ("FAIL: abs top bit type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%s good=%d res=%d\n", n,
                                      m.s[n] != 0 ? "min" : "0",
                                      (int)(m.s[n] != 0), (int)res_top.s[n]);
                            }
                          error = true;
                        }
                    }))

kernel void
test_abs ()
{
  CALL_FUNC_G (test_abs)
}
