#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_add_sat, ({
                      // add_sat
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal
                              = equal && res_add_sat.s[n] == good_add_sat.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: add_sat type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d b=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)y.s[n],
                                      (int)good_add_sat.s[n],
                                      (int)res_add_sat.s[n]);
                            }
                          error = true;
                        }
                    }))

kernel void
test_add_sat ()
{
  CALL_FUNC_G (test_add_sat)
}
