#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_abs_diff, ({
                      // abs_diff
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal
                                  && res_abs_diff.s[n] == good_abs_diff.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: abs_diff type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d b=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)y.s[n],
                                      (int)good_abs_diff.s[n],
                                      (int)res_abs_diff.s[n]);
                            }
                          error = true;
                        }
                    }))

kernel void
test_abs_diff ()
{
  CALL_FUNC_G (test_abs_diff)
}
