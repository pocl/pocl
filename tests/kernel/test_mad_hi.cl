#include "common.cl"
#include "common_hadd.cl"

DEFINE_BODY_G_HADD (test_mad_hi, ({
                      // mad_hi
                      equal = true;
                      for (int n = 0; n < vecsize; ++n)
                        {
                          equal = equal && res_mad_hi.s[n] == good_mad_hi.s[n];
                        }
                      if (!equal)
                        {
                          printf ("FAIL: mad_hi type=%s\n", typename);
                          for (int n = 0; n < vecsize; ++n)
                            {
                              printf (
                                  "   [%d] a=%d b=%d c=%d good=%d res=%d\n", n,
                                  (int)x.s[n], (int)y.s[n], (int)z.s[n],
                                  (int)good_mad_hi.s[n], (int)res_mad_hi.s[n]);
                            }
                          error = true;
                        }
                    }))

kernel void
test_mad_hi ()
{
  CALL_FUNC_G (test_mad_hi)
}
