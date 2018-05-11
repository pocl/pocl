// TESTING: <<
// TESTING: >>
// TESTING: rotate

#include "common.cl"

DEFINE_BODY_G
(test_rotate,
 ({
   int patterns[] = {0x01, 0x80, 0x77, 0xee};
   for (int p=0; p<4; ++p) {
     int const bits = count_bits(sgtype);
     int array[64];
     for (int b=0; b<bits; ++b) {
       array[b] = !!(patterns[p] & (1 << (b & 7)));
     }
     typedef union {
       gtype  v;
       sgtype s[16];
     } Tvec;
     for (int shiftbase=0; shiftbase<=bits; ++shiftbase) {
       for (int shiftoffset=0; shiftoffset<(vecsize==1?1:4); ++shiftoffset) {
         Tvec shift;
         Tvec val;
         Tvec shl, shr, rot;
         for (int n=0; n<vecsize; ++n) {
           shift.s[n] = shiftbase + n*shiftoffset;
           // Scalar shift operations undergo integer promotion, i.e.
           // the arguments are converted to int if they are smaller
           // than int. Therefore, overflowing shift counts are
           // interpreted differently.
           if ((vecsize==1) & (sizeof(sgtype) < sizeof(int))) {
             shift.s[n] &= bits-1;
           }
           val.s[n] = 0;
           shl.s[n] = 0;
           shr.s[n] = 0;
           rot.s[n] = 0;
           for (int b=bits; b>=0; --b) {
             int bl = b - (shift.s[n] & (bits-1));
             int br = b + (shift.s[n] & (bits-1));
             int sign = is_signed(sgtype) ? array[bits-1] : 0;
             val.s[n] = (val.s[n] << 1) | array[b];
             shl.s[n] = (shl.s[n] << 1) | (bl < 0 ? 0 : array[bl]);
             shr.s[n] = (shr.s[n] << 1) | (br >= bits ? sign : array[br]);
             rot.s[n] = (rot.s[n] << 1) | array[bl & (bits-1)];
           }
         }
         Tvec res;
         bool equal;
         /* shift left */
         res.v = val.v << shift.v;
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal && res.s[n] == shl.s[n];
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: shift left (<<) type=%s pattern=0x%x shiftbase=%d shiftoffset=%d val=0x%08x count=0x%08x res=0x%08x good=0x%08x\n",
                    typename, patterns[p], shiftbase, shiftoffset,
                    (uint)val.s[n], (uint)shift.s[n],
                    (uint)res.s[n], (uint)shl.s[n]);
           }
           return;
         }
         /* shift right */
         res.v = val.v >> shift.v;
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal && res.s[n] == shr.s[n];
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: shift right (>>) type=%s pattern=0x%x shiftbase=%d shiftoffset=%d val=0x%08x count=0x%08x res=0x%08x good=0x%08x\n",
                    typename, patterns[p], shiftbase, shiftoffset,
                    (uint)val.s[n], (uint)shift.s[n],
                    (uint)res.s[n], (uint)shr.s[n]);
           }
           return;
         }
         /* rotate */
         res.v = rotate(val.v, shift.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal && res.s[n] == rot.s[n];
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: rotate type=%s pattern=0x%x shiftbase=%d shiftoffset=%d val=0x%08x count=0x%08x res=0x%08x good=0x%08x\n",
                    typename, patterns[p], shiftbase, shiftoffset,
                    (uint)val.s[n], (uint)shift.s[n],
                    (uint)res.s[n], (uint)rot.s[n]);
           }
           return;
         }
       }
     }
   }
 })
 )

kernel void test_rotate()
{
  CALL_FUNC_G(test_rotate)
}
