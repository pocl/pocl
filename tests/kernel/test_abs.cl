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
                    }))

kernel void
test_abs ()
{
  CALL_FUNC_G (test_abs)
}
