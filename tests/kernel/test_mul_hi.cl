#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_mul_hi, ({
                      // mul_hi
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal && res_mul_hi.s[n] == good_mul_hi.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: mul_hi type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf ("   [%d] a=%d b=%d good=%d res=%d\n", n,
                                      (int)x.s[n], (int)y.s[n],
                                      (int)good_mul_hi.s[n],
                                      (int)res_mul_hi.s[n]);
                            }
                          error = true;
                        }
                    }))

kernel void
test_mul_hi ()
{
  CALL_FUNC_G (test_mul_hi)
}
