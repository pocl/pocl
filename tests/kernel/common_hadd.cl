/* Safe-but-slow arithmetic that can handle larger numbers without
   overflowing. */
#define DEFINE_SAFE_1(STYPE)                                                  \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_normalize (STYPE##2 const a)                 \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE##2 b;                                                               \
    b.s0 = a.s0 & halfmask;                                                   \
    b.s1 = a.s1 + (STYPE) (a.s0 >> halfbits);                                 \
    return b;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_normalize (STYPE##4 const a)                 \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE tmp;                                                                \
    STYPE##4 b;                                                               \
    tmp = a.s0;                                                               \
    b.s0 = tmp & halfmask;                                                    \
    tmp = (STYPE) (tmp >> halfbits) + a.s1;                                   \
    b.s1 = tmp & halfmask;                                                    \
    tmp = (STYPE) (tmp >> halfbits) + a.s2;                                   \
    b.s2 = tmp & halfmask;                                                    \
    tmp = (STYPE) (tmp >> halfbits) + a.s3;                                   \
    b.s3 = tmp;                                                               \
    return b;                                                                 \
  }                                                                           \
                                                                              \
  STYPE _CL_OVERLOADABLE safe_extract (STYPE##2 const a)                      \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE b;                                                                  \
    b = a.s0 | (STYPE) (a.s1 << halfbits);                                    \
    return b;                                                                 \
  }                                                                           \
                                                                              \
  STYPE _CL_OVERLOADABLE safe_extract (STYPE##4 const a)                      \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE b;                                                                  \
    if (safe_extract (a.hi) != 0 && safe_extract (a.hi) != -1)                \
      {                                                                       \
        printf ("FAIL: safe_extract [%d,%d,%d,%d]\n", (int)a.s0, (int)a.s1,   \
                (int)a.s2, (int)a.s3);                                        \
      }                                                                       \
    return safe_extract (a.lo);                                               \
  }                                                                           \
                                                                              \
  bool _CL_OVERLOADABLE safe_isneg (STYPE##2 a) { return a.s1 < (STYPE)0; }   \
                                                                              \
  bool _CL_OVERLOADABLE safe_isneg (STYPE##4 a) { return a.s3 < (STYPE)0; }   \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_neg (STYPE##2 a)                             \
  {                                                                           \
    STYPE##2 b;                                                               \
    b.s0 = -a.s0;                                                             \
    b.s1 = -a.s1;                                                             \
    return safe_normalize (b);                                                \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_neg (STYPE##4 a)                             \
  {                                                                           \
    STYPE##4 b;                                                               \
    b.s0 = -a.s0;                                                             \
    b.s1 = -a.s1;                                                             \
    b.s2 = -a.s2;                                                             \
    b.s3 = -a.s3;                                                             \
    return safe_normalize (b);                                                \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_abs (STYPE##2 const a)                       \
  {                                                                           \
    STYPE##2 b;                                                               \
    b = a;                                                                    \
    if (safe_isneg (b))                                                       \
      {                                                                       \
        b = safe_neg (b);                                                     \
      }                                                                       \
    return b;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_abs (STYPE##4 const a)                       \
  {                                                                           \
    STYPE##4 b;                                                               \
    b = a;                                                                    \
    if (safe_isneg (b))                                                       \
      {                                                                       \
        b = safe_neg (b);                                                     \
      }                                                                       \
    return b;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_add (STYPE##2 const a, STYPE##2 const b)     \
  {                                                                           \
    STYPE##2 c;                                                               \
    c.s0 = a.s0 + b.s0;                                                       \
    c.s1 = a.s1 + b.s1;                                                       \
    return safe_normalize (c);                                                \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_add (STYPE##4 const a, STYPE##4 const b)     \
  {                                                                           \
    STYPE##4 c;                                                               \
    c.s0 = a.s0 + b.s0;                                                       \
    c.s1 = a.s1 + b.s1;                                                       \
    c.s2 = a.s2 + b.s2;                                                       \
    c.s3 = a.s3 + b.s3;                                                       \
    return safe_normalize (c);                                                \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_sub (STYPE##2 const a, STYPE##2 const b)     \
  {                                                                           \
    STYPE##2 c;                                                               \
    c.s0 = a.s0 - b.s0;                                                       \
    c.s1 = a.s1 - b.s1;                                                       \
    return safe_normalize (c);                                                \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_sub (STYPE##4 const a, STYPE##4 const b)     \
  {                                                                           \
    STYPE##4 c;                                                               \
    c.s0 = a.s0 - b.s0;                                                       \
    c.s1 = a.s1 - b.s1;                                                       \
    c.s2 = a.s2 - b.s2;                                                       \
    c.s3 = a.s3 - b.s3;                                                       \
    return safe_normalize (c);                                                \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_create (STYPE const a);                      \
  STYPE##2 _CL_OVERLOADABLE safe_minimul (STYPE const a, STYPE const b)       \
  {                                                                           \
    STYPE##2 tmp1 = safe_create ((STYPE) (a * (STYPE) (b & (STYPE)1)));       \
    STYPE##2 tmp2 = safe_create ((STYPE) (a * (STYPE) (b >> (STYPE)1)));      \
    STYPE##2 res;                                                             \
    res = safe_add (tmp1, safe_add (tmp2, tmp2));                             \
    return res;                                                               \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_mul (STYPE##2 a, STYPE##2 b)                 \
  {                                                                           \
    bool a_neg = safe_isneg (a);                                              \
    bool b_neg = safe_isneg (b);                                              \
    a = safe_abs (a);                                                         \
    b = safe_abs (b);                                                         \
    STYPE##4 c00, c01, c10, c11;                                              \
    c00 = 0;                                                                  \
    c00.s01 = safe_minimul (a.s0, b.s0);                                      \
    c00 = safe_normalize (c00);                                               \
    c01 = 0;                                                                  \
    c01.s12 = safe_minimul (a.s0, b.s1);                                      \
    c01 = safe_normalize (c01);                                               \
    c10 = 0;                                                                  \
    c10.s12 = safe_minimul (a.s1, b.s0);                                      \
    c10 = safe_normalize (c10);                                               \
    c11 = 0;                                                                  \
    c11.s23 = safe_minimul (a.s1, b.s1);                                      \
    c11 = safe_normalize (c11);                                               \
    STYPE##4 c;                                                               \
    c = safe_add (safe_add (c00, c01), safe_add (c10, c11));                  \
    if (a_neg ^ b_neg)                                                        \
      c = safe_neg (c);                                                       \
    return c;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_max (STYPE##2 const a, STYPE##2 const b)     \
  {                                                                           \
    STYPE##2 c;                                                               \
    if (safe_isneg (safe_sub (a, b)))                                         \
      {                                                                       \
        c = b;                                                                \
      }                                                                       \
    else                                                                      \
      {                                                                       \
        c = a;                                                                \
      }                                                                       \
    return c;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_max (STYPE##4 const a, STYPE##4 const b)     \
  {                                                                           \
    STYPE##4 c;                                                               \
    if (safe_isneg (safe_sub (a, b)))                                         \
      {                                                                       \
        c = b;                                                                \
      }                                                                       \
    else                                                                      \
      {                                                                       \
        c = a;                                                                \
      }                                                                       \
    return c;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_min (STYPE##2 const a, STYPE##2 const b)     \
  {                                                                           \
    STYPE##2 c;                                                               \
    if (safe_isneg (safe_sub (a, b)))                                         \
      {                                                                       \
        c = a;                                                                \
      }                                                                       \
    else                                                                      \
      {                                                                       \
        c = b;                                                                \
      }                                                                       \
    return c;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_min (STYPE##4 const a, STYPE##4 const b)     \
  {                                                                           \
    STYPE##4 c;                                                               \
    if (safe_isneg (safe_sub (a, b)))                                         \
      {                                                                       \
        c = a;                                                                \
      }                                                                       \
    else                                                                      \
      {                                                                       \
        c = b;                                                                \
      }                                                                       \
    return c;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_clamp (STYPE##2 const a, STYPE##2 const alo, \
                                        STYPE##2 const ahi)                   \
  {                                                                           \
    return safe_max (alo, safe_min (ahi, a));                                 \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_clamp (STYPE##4 const a, STYPE##4 const alo, \
                                        STYPE##4 const ahi)                   \
  {                                                                           \
    return safe_max (alo, safe_min (ahi, a));                                 \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_rshift (STYPE##2 a)                          \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE##2 b;                                                               \
    b.s0 = a.s0 | ((a.s1 & (STYPE)1) << halfbits);                            \
    b.s1 = a.s1 & ~(STYPE)1;                                                  \
    b.s0 >>= (STYPE)1;                                                        \
    b.s1 >>= (STYPE)1;                                                        \
    return safe_normalize (b);                                                \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_lo (STYPE##2 a)                              \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    bool a_neg = a.s1 < (STYPE)0;                                             \
    a = safe_abs (a);                                                         \
    if (a.s1 >= halfmax)                                                      \
      a.s1 &= halfmask;                                                       \
    if (a_neg)                                                                \
      a = safe_neg (a);                                                       \
    return a;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_lo (STYPE##4 a)                              \
  {                                                                           \
    bool a_neg = a.s3 < (STYPE)0;                                             \
    a = safe_abs (a);                                                         \
    STYPE##2 res = safe_normalize (a.lo);                                     \
    if (a_neg)                                                                \
      res = safe_neg (res);                                                   \
    return res;                                                               \
  }                                                                           \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_hi (STYPE##4 a)                              \
  {                                                                           \
    return safe_normalize (a.hi);                                             \
  }

#define DEFINE_SAFE_2(TYPE, STYPE)                                            \
                                                                              \
  STYPE##2 _CL_OVERLOADABLE safe_create (TYPE const a)                        \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE##2 b;                                                               \
    /* input may be unsigned */                                               \
    b.s0 = a & (TYPE)halfmask;                                                \
    b.s1 = a >> (TYPE)halfbits;                                               \
    b = safe_normalize (b);                                                   \
    if ((TYPE)safe_extract (b) != a)                                          \
      {                                                                       \
        printf ("FAIL: safe_create %d (got %d)\n", (int)a,                    \
                (int)(TYPE)safe_extract (b));                                 \
      }                                                                       \
    return b;                                                                 \
  }                                                                           \
                                                                              \
  STYPE##4 _CL_OVERLOADABLE safe_create4 (TYPE const a)                       \
  {                                                                           \
    STYPE const halfbits = 4 * sizeof (STYPE);                                \
    STYPE const halfmax = (STYPE)1 << halfbits;                               \
    STYPE const halfmask = halfmax - (STYPE)1;                                \
    STYPE##4 b;                                                               \
    /* input may be unsigned */                                               \
    TYPE tmp = a;                                                             \
    b.s0 = tmp & (TYPE)halfmask;                                              \
    tmp >>= halfbits;                                                         \
    b.s1 = tmp & (TYPE)halfmask;                                              \
    tmp >>= halfbits;                                                         \
    b.s2 = tmp & (TYPE)halfmask;                                              \
    tmp >>= halfbits;                                                         \
    b.s3 = tmp;                                                               \
    b = safe_normalize (b);                                                   \
    if ((TYPE)safe_extract (b) != a)                                          \
      {                                                                       \
        printf ("FAIL: safe_create4 sz=%d sg=%d %d (got %d) [%d,%d,%d,%d]\n", \
                (int)sizeof (TYPE), (int)((TYPE)-1 < (TYPE)0), (int)a,        \
                (int)(TYPE)safe_extract (b), (int)b.s0, (int)b.s1, (int)b.s2, \
                (int)b.s3);                                                   \
      }                                                                       \
    return b;                                                                 \
  }



DEFINE_SAFE_1 (char)
DEFINE_SAFE_1 (short)
DEFINE_SAFE_1 (int)
__IF_INT64 (DEFINE_SAFE_1 (long))

DEFINE_SAFE_2 (char, char)
DEFINE_SAFE_2 (uchar, char)
DEFINE_SAFE_2 (short, short)
DEFINE_SAFE_2 (ushort, short)
DEFINE_SAFE_2 (int, int)
DEFINE_SAFE_2 (uint, int)
__IF_INT64 (DEFINE_SAFE_2 (long, long))
__IF_INT64 (DEFINE_SAFE_2 (ulong, long))



#define IMPLEMENT_BODY_G_HADD(NAME, BODY, SIZE, GTYPE, SGTYPE, UGTYPE,        \
                              SUGTYPE)                                        \
  void NAME##_##GTYPE ()                                                      \
  {                                                                           \
    typedef GTYPE gtype;                                                      \
    typedef SGTYPE sgtype;                                                    \
    typedef UGTYPE ugtype;                                                    \
    typedef SUGTYPE sugtype;                                                  \
    string const typename = #GTYPE;                                           \
    const int vecsize = SIZE;                                                 \
    int const bits = count_bits (sgtype);                                     \
    sgtype const tmin = is_signed (sgtype)                                    \
                            ? (sgtype) ((sugtype)1 << (sugtype) (bits - 1))   \
                            : (sgtype)0;                                      \
    sgtype const tmax = (sgtype) ((sugtype)tmin - (sugtype)1);                \
    for (int iter = 0; iter < nrandoms; ++iter)                               \
      {                                                                       \
        typedef union                                                         \
        {                                                                     \
          gtype v;                                                            \
          ugtype u;                                                           \
          sgtype s[16];                                                       \
        } Tvec;                                                               \
        Tvec x, y, z;                                                         \
        Tvec good_abs;                                                        \
        Tvec good_abs_diff, good_add_sat, good_mad_sat, good_sub_sat;         \
        Tvec good_hadd, good_mad_hi, good_mul_hi, good_rhadd;                 \
        for (int n = 0; n < vecsize; ++n)                                     \
          {                                                                   \
            x.s[n] = randoms[(iter + n) % nrandoms];                          \
            y.s[n] = randoms[(iter + n + 20) % nrandoms];                     \
            z.s[n] = randoms[(iter + n + 40) % nrandoms];                     \
            if (bits > 32)                                                    \
              {                                                               \
                x.s[n] = (x.s[n] << (bits / 2))                               \
                         | randoms[(iter + n + 100) % nrandoms];              \
                y.s[n] = (y.s[n] << (bits / 2))                               \
                         | randoms[(iter + n + 120) % nrandoms];              \
                z.s[n] = (z.s[n] << (bits / 2))                               \
                         | randoms[(iter + n + 140) % nrandoms];              \
              }                                                               \
            good_abs.s[n] = safe_extract (safe_abs (safe_create (x.s[n])));   \
            good_abs_diff.s[n] = safe_extract (safe_abs (                     \
                safe_sub (safe_create (x.s[n]), safe_create (y.s[n]))));      \
            good_add_sat.s[n] = safe_extract (safe_clamp (                    \
                safe_add (safe_create (x.s[n]), safe_create (y.s[n])),        \
                safe_create (tmin), safe_create (tmax)));                     \
            good_mad_sat.s[n] = safe_extract (                                \
                safe_clamp (safe_add (safe_mul (safe_create (x.s[n]),         \
                                                safe_create (y.s[n])),        \
                                      safe_create4 (z.s[n])),                 \
                            safe_create4 (tmin), safe_create4 (tmax)));       \
            good_sub_sat.s[n] = safe_extract (safe_clamp (                    \
                safe_sub (safe_create (x.s[n]), safe_create (y.s[n])),        \
                safe_create (tmin), safe_create (tmax)));                     \
            good_hadd.s[n] = safe_extract (safe_rshift (                      \
                safe_add (safe_create (x.s[n]), safe_create (y.s[n]))));      \
            good_mad_hi.s[n] = safe_extract (                                 \
                safe_lo (safe_add (safe_hi (safe_mul (safe_create (x.s[n]),   \
                                                      safe_create (y.s[n]))), \
                                   safe_create (z.s[n]))));                   \
            good_mul_hi.s[n] = safe_extract (safe_hi (                        \
                safe_mul (safe_create (x.s[n]), safe_create (y.s[n]))));      \
            good_rhadd.s[n] = safe_extract (safe_rshift (safe_add (           \
                safe_add (safe_create (x.s[n]), safe_create (y.s[n])),        \
                safe_create ((sgtype)1))));                                   \
          }                                                                   \
        Tvec res_abs;                                                         \
        Tvec res_abs_diff, res_add_sat, res_mad_sat, res_sub_sat;             \
        Tvec res_hadd, res_mad_hi, res_mul_hi, res_rhadd;                     \
        res_abs.u = abs (x.v);                                                \
        res_abs_diff.u = abs_diff (x.v, y.v);                                 \
        res_add_sat.v = add_sat (x.v, y.v);                                   \
        res_mad_sat.v = mad_sat (x.v, y.v, z.v);                              \
        res_sub_sat.v = sub_sat (x.v, y.v);                                   \
        res_hadd.v = hadd (x.v, y.v);                                         \
        res_mad_hi.v = mad_hi (x.v, y.v, z.v);                                \
        res_mul_hi.v = mul_hi (x.v, y.v);                                     \
        res_rhadd.v = rhadd (x.v, y.v);                                       \
        bool error = false;                                                   \
        bool equal;                                                           \
        BODY;                                                                 \
      }                                                                       \
  }


#define DEFINE_BODY_G_HADD(NAME, EXPR)                                        \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, char, char, uchar, uchar)             \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, char2, char, uchar2, uchar)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, char3, char, uchar3, uchar)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, char4, char, uchar4, uchar)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, char8, char, uchar8, uchar)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, char16, char, uchar16, uchar)        \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, uchar, uchar, uchar, uchar)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, uchar2, uchar, uchar2, uchar)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, uchar3, uchar, uchar3, uchar)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, uchar4, uchar, uchar4, uchar)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, uchar8, uchar, uchar8, uchar)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, uchar16, uchar, uchar16, uchar)      \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, short, short, ushort, ushort)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, short2, short, ushort2, ushort)       \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, short3, short, ushort3, ushort)       \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, short4, short, ushort4, ushort)       \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, short8, short, ushort8, ushort)       \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, short16, short, ushort16, ushort)    \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, ushort, ushort, ushort, ushort)       \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, ushort2, ushort, ushort2, ushort)     \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, ushort3, ushort, ushort3, ushort)     \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, ushort4, ushort, ushort4, ushort)     \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, ushort8, ushort, ushort8, ushort)     \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, ushort16, ushort, ushort16, ushort)  \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, int, int, uint, uint)                 \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, int2, int, uint2, uint)               \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, int3, int, uint3, uint)               \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, int4, int, uint4, uint)               \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, int8, int, uint8, uint)               \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, int16, int, uint16, uint)            \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, uint, uint, uint, uint)               \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, uint2, uint, uint2, uint)             \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, uint3, uint, uint3, uint)             \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, uint4, uint, uint4, uint)             \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, uint8, uint, uint8, uint)             \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, uint16, uint, uint16, uint)          \
  __IF_INT64 (                                                                \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, long, long, ulong, ulong)             \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, long2, long, ulong2, ulong)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, long3, long, ulong3, ulong)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, long4, long, ulong4, ulong)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, long8, long, ulong8, ulong)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, long16, long, ulong16, ulong)        \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 1, ulong, ulong, ulong, ulong)           \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 2, ulong2, ulong, ulong2, ulong)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 3, ulong3, ulong, ulong3, ulong)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 4, ulong4, ulong, ulong4, ulong)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 8, ulong8, ulong, ulong8, ulong)         \
  IMPLEMENT_BODY_G_HADD (NAME, EXPR, 16, ulong16, ulong, ulong16, ulong))
