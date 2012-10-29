// TESTING: <<
// TESTING: >>
// TESTING: rotate

#define IMPLEMENT_BODY_G(NAME, BODY, GTYPE, SGTYPE, UGTYPE, SUGTYPE)    \
  void NAME##_##GTYPE()                                                 \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    char const *const typename = #GTYPE;                                \
    BODY;                                                               \
  }
#define DEFINE_BODY_G(NAME, EXPR)                                       \
  IMPLEMENT_BODY_G(NAME, EXPR, char    , char  , uchar   , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, char2   , char  , uchar2  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, char3   , char  , uchar3  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, char4   , char  , uchar4  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, char8   , char  , uchar8  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, char16  , char  , uchar16 , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uchar   , uchar , uchar   , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, short   , short , ushort  , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, short2  , short , ushort2 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, short3  , short , ushort3 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, short4  , short , ushort4 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, short8  , short , ushort8 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, short16 , short , ushort16, ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, ushort  , ushort, ushort  , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, ushort16, ushort, ushort16, ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, int     , int   , uint    , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, int2    , int   , uint2   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, int3    , int   , uint3   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, int4    , int   , uint4   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, int8    , int   , uint8   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, int16   , int   , uint16  , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uint    , uint  , uint    , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uint2   , uint  , uint2   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uint3   , uint  , uint3   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uint4   , uint  , uint4   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uint8   , uint  , uint8   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, uint16  , uint  , uint16  , uint  )      \
  __IF_INT64(                                                           \
  IMPLEMENT_BODY_G(NAME, EXPR, long    , long  , ulong   , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, long2   , long  , ulong2  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, long3   , long  , ulong3  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, long4   , long  , ulong4  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, long8   , long  , ulong8  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, long16  , long  , ulong16 , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, ulong   , ulong , ulong   , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define CALL_FUNC_G(NAME)                       \
  NAME##_char    ();                            \
  NAME##_char2   ();                            \
  NAME##_char3   ();                            \
  NAME##_char4   ();                            \
  NAME##_char8   ();                            \
  NAME##_char16  ();                            \
  NAME##_uchar   ();                            \
  NAME##_uchar2  ();                            \
  NAME##_uchar3  ();                            \
  NAME##_uchar4  ();                            \
  NAME##_uchar8  ();                            \
  NAME##_uchar16 ();                            \
  NAME##_short   ();                            \
  NAME##_short2  ();                            \
  NAME##_short3  ();                            \
  NAME##_short4  ();                            \
  NAME##_short8  ();                            \
  NAME##_short16 ();                            \
  NAME##_ushort  ();                            \
  NAME##_ushort2 ();                            \
  NAME##_ushort3 ();                            \
  NAME##_ushort4 ();                            \
  NAME##_ushort8 ();                            \
  NAME##_ushort16();                            \
  NAME##_int     ();                            \
  NAME##_int2    ();                            \
  NAME##_int3    ();                            \
  NAME##_int4    ();                            \
  NAME##_int8    ();                            \
  NAME##_int16   ();                            \
  NAME##_uint    ();                            \
  NAME##_uint2   ();                            \
  NAME##_uint3   ();                            \
  NAME##_uint4   ();                            \
  NAME##_uint8   ();                            \
  NAME##_uint16  ();                            \
  __IF_INT64(                                   \
  NAME##_long    ();                            \
  NAME##_long2   ();                            \
  NAME##_long3   ();                            \
  NAME##_long4   ();                            \
  NAME##_long8   ();                            \
  NAME##_long16  ();                            \
  NAME##_ulong   ();                            \
  NAME##_ulong2  ();                            \
  NAME##_ulong3  ();                            \
  NAME##_ulong4  ();                            \
  NAME##_ulong8  ();                            \
  NAME##_ulong16 ();)



#define is_signed(T)   ((T)-1 < (T)+1)
#define is_floating(T) ((T)0.1f > (T)0.0f)
#define count_bits(T)  (CHAR_BIT * sizeof(T))

DEFINE_BODY_G
(test_rotate,
 ({
   _cl_static_assert(sgtype, !is_floating(sgtype));
   int patterns[] = {0x01, 0x80, 0x77, 0xee};
   for (int p=0; p<4; ++p) {
     int const bits = count_bits(sgtype);
     int array[bits];
     for (int b=0; b<bits; ++b) {
       array[b] = !!(patterns[p] & (1 << (b & 7)));
     }
     int vecsize = vec_step(gtype);
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
           val.s[n]  = 0;
           shl.s[n]  = 0;
           shr.s[n]  = 0;
           rot.s[n]  = 0;
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
           printf("FAIL: shift left (<<) type=%s pattern=0x%x shiftbase=%d shiftoffset=%d res=0x%08x good=0x%08x\n",
                  typename, patterns[p], shiftbase, shiftoffset,
                  (uint)res.s[0], (uint)shl.s[0]);
           return;
         }
         /* shift right */
         res.v = val.v >> shift.v;
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal && res.s[n] == shr.s[n];
         }
         if (!equal) {
           printf("FAIL: shift right (>>) type=%s pattern=0x%x shiftbase=%d shiftoffset=%d res=0x%08x good=0x%08x\n",
                  typename, patterns[p], shiftbase, shiftoffset,
                  (uint)res.s[0], (uint)shr.s[0]);
           return;
         }
         /* rotate */
         res.v = rotate(val.v, shift.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal && res.s[n] == rot.s[n];
         }
         if (!equal) {
           printf("FAIL: rotate type=%s pattern=0x%x shiftbase=%d shiftoffset=%d res=0x%08x good=0x%08x\n",
                  typename, patterns[p], shiftbase, shiftoffset,
                  (uint)res.s[0], (uint)rot.s[0]);
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
