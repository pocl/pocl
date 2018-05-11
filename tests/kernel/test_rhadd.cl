#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_rhadd, ({
                      // rhadd
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal && res_rhadd.s[n] == good_rhadd.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: rhadd type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d b=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)y.s[n],
                                      (int)good_rhadd.s[n],
                                      (int)res_rhadd.s[n]);
                            }
                          error = true;
                        }
                    }))

kernel void
test_rhadd ()
{
  CALL_FUNC_G (test_rhadd)
}
