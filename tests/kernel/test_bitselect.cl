// TESTING: abs
// TESTING: bitselect
// TESTING: clz
// TESTING: max
// TESTING: min
// TESTING: popcount

#include "common.cl"


DEFINE_BODY_G(test_bitselect, ({
   int const bits = count_bits(sgtype);
   for (int iter=0; iter<nrandoms; ++iter) {
     typedef union {
       gtype  v;
       sgtype s[16];
     } Tvec;
     typedef union {
       ugtype  v;
       sugtype s[16];
     } UTvec;
     Tvec sel, left, right;
     UTvec res_abs;
     Tvec res_bitselect, res_clz, res_max, res_min, res_popcount;
     for (int n=0; n<vecsize; ++n) {
       sel.s[n]   = randoms[(iter+n   ) % nrandoms];
       left.s[n]  = randoms[(iter+n+20) % nrandoms];
       right.s[n] = randoms[(iter+n+40) % nrandoms];
       if (bits>32) {
         sel.s[n]   = (sel.s[n]   << (bits/2)) | randoms[(iter+n+100) % nrandoms];
         left.s[n]  = (left.s[n]  << (bits/2)) | randoms[(iter+n+120) % nrandoms];
         right.s[n] = (right.s[n] << (bits/2)) | randoms[(iter+n+140) % nrandoms];
       }
     }
     res_abs.v = abs(left.v);
     res_bitselect.v = bitselect(left.v, right.v, sel.v);
     res_clz.v = clz(left.v);
     res_max.v = max(left.v, right.v);
     res_min.v = min(left.v, right.v);
     res_popcount.v = popcount(left.v);
     bool equal;
     // abs
     equal = true;
     for (int n=0; n<vecsize; ++n) {
       sgtype signbit = (sgtype)((sgtype)1 << (sgtype)(count_bits(sgtype)-1));
       // Note: left.s[n] < 0 leads to a compiler warning for unsigned types,
       // so we check the sign bit explicitly
       sugtype absval =
         is_signed(sgtype) ?
         (left.s[n] & signbit ? -left.s[n] : left.s[n]) :
         left.s[n];
       if (res_abs.s[n] != absval) {
         equal = false;
         printf("FAIL: abs(a)[%d] type=%s a=0x%08x want=0x%08x got=0x%08x\n",
                n, typename,
                (uint)left.s[n], (uint)absval,
                (uint)res_abs.s[n]);
       }
     }
     // bitselect
     for (int n=0; n<vecsize; ++n) {
       sgtype selval = (left.s[n] & ~sel.s[n]) | (right.s[n] & sel.s[n]);
       if (res_bitselect.s[n] != selval) {
         equal = false;
         printf("FAIL: bitselect(a,b,c)[%d] type=%s a=0x%08x b=0x%08x c=0x%08x want=0x%08x got=0x%08x\n",
                n, typename,
                (uint)left.s[n], (uint)right.s[n], (uint)sel.s[n], (uint)selval,
                (uint)res_bitselect.s[n]);
       }
     }
     // clz
     for (int n=0; n<vecsize; ++n) {
       int b=0;
       while (b<bits) {
         sgtype mask = (sgtype)1 << (sgtype)(bits - 1 - b);
         if (left.s[n] & mask) break;
         ++b;
       }
       if (res_clz.s[n] != (sgtype)b) {
         equal = false;
         printf("FAIL: clz(a)[%d] type=%s a=0x%08x want=0x%08x got=0x%08x\n",
                n, typename,
                (uint)left.s[n], (uint)b,
                (uint)res_clz.s[n]);
       }
     }
     // max
     for (int n=0; n<vecsize; ++n) {
       sgtype maxval = left.s[n] > right.s[n] ? left.s[n] : right.s[n];
       if (res_max.s[n] != maxval) {
         equal = false;
         printf("FAIL: max(a,b)[%d] type=%s a=0x%08x b=0x%08x want=0x%08x got=0x%08x\n",
                n, typename,
                (uint)left.s[n], (uint)right.s[n], (uint)maxval,
                (uint)res_max.s[n]);
       }
     }
     // min
     for (int n=0; n<vecsize; ++n) {
       sgtype minval = left.s[n] < right.s[n] ? left.s[n] : right.s[n];
       if (res_min.s[n] != minval) {
         equal = false;
         printf("FAIL: min(a,b)[%d] type=%s a=0x%08x b=0x%08x want=0x%08x got=0x%08x\n",
                n, typename,
                (uint)left.s[n], (uint)right.s[n], (uint)minval,
                (uint)res_min.s[n]);
       }
     }
     // popcount
     for (int n=0; n<vecsize; ++n) {
       int c=0;
       for (int b=0; b<bits; ++b) {
         sgtype mask = (sgtype)1 << (sgtype)b;
         if (left.s[n] & mask) ++c;
       }
       if (res_popcount.s[n] != (sgtype)c) {
         equal = false;
         printf("FAIL: popcount(a)[%d] type=%s a=0x%08x want=0x%08x got=0x%08x\n",
                n, typename,
                (uint)left.s[n], (uint)c,
                (uint)res_clz.s[n]);
       }
     }
     if (!equal) return;
   }
 })
)

kernel void test_bitselect()
{
  CALL_FUNC_G(test_bitselect)
}
