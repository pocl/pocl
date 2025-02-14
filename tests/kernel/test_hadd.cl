#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_hadd, ({
                      // hadd
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal && res_hadd.s[n] == good_hadd.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: hadd type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d b=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)y.s[n],
                                      (int)good_hadd.s[n], (int)res_hadd.s[n]);
                            }
                          error = true;
                        }
                    }))

DEF_KERNELS_CALL_FUNC_G (test_hadd);
