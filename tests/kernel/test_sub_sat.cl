#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_sub_sat, ({
                      // sub_sat
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal
                              = equal && res_sub_sat.s[n] == good_sub_sat.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: sub_sat type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d b=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)y.s[n],
                                      (int)good_sub_sat.s[n],
                                      (int)res_sub_sat.s[n]);
                            }
                          error = true;
                        }
                      if (error)
                        return;
                    }))

kernel void
test_sub_sat ()
{
  CALL_FUNC_G (test_sub_sat)
}
