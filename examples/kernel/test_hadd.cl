// TESTING: abs
// TESTING: abs_diff
// TESTING: add_sat
// TESTING: hadd
// TESTING: rhadd



/* Safe-but-slow arithmetic that can handle larger numbers without
   overflowing. */
#define DEFINE_SAFE_1(STYPE)                                            \
                                                                        \
  STYPE##2 _cl_overloadable safe_normalize(STYPE##2 const a)            \
  {                                                                     \
    STYPE const halfbits = 4*sizeof(STYPE);                             \
    STYPE const halfmax = (STYPE)1 << halfbits;                         \
    STYPE const halfmask = halfmax - (STYPE)1;                          \
    STYPE##2 b;                                                         \
    b.s0 = a.s0 & halfmask;                                             \
    b.s1 = a.s1 + (a.s0 >> halfbits);                                   \
    return b;                                                           \
  }                                                                     \
                                                                        \
  STYPE _cl_overloadable safe_extract(STYPE##2 const a)                 \
  {                                                                     \
    STYPE const halfbits = 4*sizeof(STYPE);                             \
    STYPE const halfmax = (STYPE)1 << halfbits;                         \
    STYPE const halfmask = halfmax - (STYPE)1;                          \
    STYPE b;                                                            \
    b = a.s0 | a.s1 << halfbits;                                        \
    return b;                                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_neg(STYPE##2 a)                        \
  {                                                                     \
    STYPE##2 b;                                                         \
    b.s0 = - a.s0;                                                      \
    b.s1 = - a.s1;                                                      \
    return safe_normalize(b);                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_abs(STYPE##2 const a)                  \
  {                                                                     \
    STYPE##2 b;                                                         \
    b = a;                                                              \
    if (b.s1 < (STYPE)0) {                                              \
      b = safe_neg(b);                                                  \
    }                                                                   \
    return b;                                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_add(STYPE##2 const a, STYPE##2 const b) \
  {                                                                     \
    STYPE##2 c;                                                         \
    c.s0 = a.s0 + b.s0;                                                 \
    c.s1 = a.s1 + b.s1;                                                 \
    return safe_normalize(c);                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_sub(STYPE##2 const a, STYPE##2 const b) \
  {                                                                     \
    STYPE##2 c;                                                         \
    c.s0 = a.s0 - b.s0;                                                 \
    c.s1 = a.s1 - b.s1;                                                 \
    return safe_normalize(c);                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_max(STYPE##2 const a, STYPE##2 const b) \
  {                                                                     \
    STYPE##2 c;                                                         \
    if (a.s1 > b.s1 || (a.s1 == b.s1 && a.s0 >= b.s0)) {                \
      c = a;                                                            \
    } else {                                                            \
      c = b;                                                            \
    }                                                                   \
    return c;                                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_min(STYPE##2 const a, STYPE##2 const b) \
  {                                                                     \
    STYPE##2 c;                                                         \
    if (a.s1 < b.s1 || (a.s1 == b.s1 && a.s0 <= b.s0)) {                \
      c = a;                                                            \
    } else {                                                            \
      c = b;                                                            \
    }                                                                   \
    return c;                                                           \
  }                                                                     \
                                                                        \
  STYPE##2 _cl_overloadable safe_rshift(STYPE##2 a)                     \
  {                                                                     \
    STYPE const halfbits = 4*sizeof(STYPE);                             \
    STYPE const halfmax = (STYPE)1 << halfbits;                         \
    STYPE const halfmask = halfmax - (STYPE)1;                          \
    STYPE##2 b;                                                         \
    b.s0 = a.s0 | ((a.s1 & (STYPE)1) << halfbits);                      \
    b.s1 = a.s1 & ~(STYPE)1;                                            \
    b.s0 >>= (STYPE)1;                                                  \
    b.s1 >>= (STYPE)1;                                                  \
    return safe_normalize(b);                                           \
  }



#define DEFINE_SAFE_2(TYPE, STYPE)                                      \
                                                                        \
  STYPE##2 _cl_overloadable safe_create(TYPE const a)                   \
  {                                                                     \
    STYPE const halfbits = 4*sizeof(STYPE);                             \
    STYPE const halfmax = (STYPE)1 << halfbits;                         \
    STYPE const halfmask = halfmax - (STYPE)1;                          \
    STYPE##2 b;                                                         \
    b.s0 = a & (TYPE)halfmask;                                          \
    b.s1 = a >> (TYPE)halfbits;                                         \
    b = safe_normalize(b);                                              \
    if ((TYPE)safe_extract(b) != a) printf("FAIL: safe_create %d\n", (int)a); \
    return b;                                                           \
  }



DEFINE_SAFE_1(char )
DEFINE_SAFE_1(short)
DEFINE_SAFE_1(int  )
__IF_INT64(
DEFINE_SAFE_1(long ))

DEFINE_SAFE_2(char  , char )
DEFINE_SAFE_2(uchar , char )
DEFINE_SAFE_2(short , short)
DEFINE_SAFE_2(ushort, short)
DEFINE_SAFE_2(int   , int  )
DEFINE_SAFE_2(uint  , int  )
__IF_INT64(
DEFINE_SAFE_2(long  , long )
DEFINE_SAFE_2(ulong , long ))



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
#define is_floating(T) ((T)0.1 > (T)0.0)
#define count_bits(T)  (CHAR_BIT * sizeof(T))

DEFINE_BODY_G
(test_hadd,
 ({
   _cl_static_assert(sgtype, !is_floating(sgtype));
   uint const randoms[] = {
     0x00000000,
     0x00000001,
     0x7fffffff,
     0x80000000,
     0xfffffffe,
     0xffffffff,
     0x01010101,
     0x80808080,
     0x55555555,
     0xaaaaaaaa,
     116127149,
     331473970,
     3314285513,
     1531519032,
     3871781304,
     723260354,
     3734992454,
     3048883544,
     424075405,
     3760586679,
     364071113,
     2212396745,
     3026460845,
     2062923368,
     3945483116,
     774301702,
     2010645213,
     353497300,
     2240089293,
     645959945,
     2929402380,
     3641106046,
     3731530029,
     3788502454,
     3990366079,
     3532452335,
     3231247251,
     123690193,
     418692672,
     4146745661,
     4170087687,
     3915754726,
     2052700648,
     1748863847,
     276568793,
     364266289,
     24718041,
     3775186845,
     935438421,
     3070232227,
     558364671,
     2318351214,
     17943242,
     1796864907,
     727165514,
     223478118,
     2448924107,
     496915291,
     3372891854,
     361433487,
     3273766229,
     251831411,
     432661417,
     772908669,
     289792578,
     4150526710,
     4157662725,
     2594757327,
     3052388893,
     3842089578,
     3467269013,
     510187125,
     2596093643,
     398042620,
     4272455984,
     3711648086,
     2120827851,
     77269246,
     2168059317,
     2750549452,
     1712682330,
     2486520097,
     625173621,
     1632501477,
     2935468416,
     980045574,
     3080136685,
     4291385683,
     1900746145,
     3343063222,
     3737266887,
     3349055009,
     3557165116,
     847440541,
     1195278641,
     313889830,
     622790046,
     326637691,
     663570370,
     662327410,
     923839117,
     3091793818,
     3563062752,
     1864236072,
     4251970867,
     2259486024,
     2512789432,
     4278284968,
     244581614,
     247706675,
     3268622648,
     3758387026,
     206893256,
     2892198447,
     3585538105,
     2484801188,
     1063964031,
     3712657639,
     23179627,
     1732005357,
     2522016557,
     1058341654,
     1580368080,
     1890361257,
     1167428989,
     2600065453,
     1547136389,
     945856727,
     2005682606,
     3399854093,
     2619154565,
     2207015138,
     2836381097,
     612928932,
     1537934908,
     897756908,
     1142275256,
     1106163744,
     3209429231,
     3317761168,
     2815958850,
     1282374282,
     3861163766,
     2547903564,
     3139840265,
     587243656,
     3261127556,
     3955999184,
     2061849860,
     3778058575,
     259659645,
     935157504,
     3294850933,
     2164603733,
     3772888022,
     732201413,
     3677934092,
     321204420,
     509807651,
     3626474557,
     284622251,
     3655952885,
     1512028769,
     1102588652,
     2700179235,
     4167405174,
     2672050627,
     3410780487,
     4153733940,
     2459759898,
     568792515,
     1081882827,
     3211871042,
     799411732,
     2101993855,
     3415550991,
     3872737342,
     4168312654,
     1889019671,
     4247531636,
     2442118552,
     3024016549,
     1041817509,
     141773691,
     28033810,
     4034097901,
     1532981240,
     2593712697,
     2751535537,
     269072724,
     3363560906,
     3555817938,
     611297346,
     366972507,
     788151801,
     3990920857,
     1611303958,
     3353102293,
     1334246396,
     1114446428,
     3491128109,
     2922751152,
     3053407478,
     2897830841,
     176546593,
     3184221063,
     37923477,
     1692128510,
     165719856,
     1795746307,
     2422422413,
     253227286,
     2188522595,
     582156087,
     2342528685,
     2080142547,
     1928462563,
     2713927482,
     1944972771,
     2534268146,
     830798003,
     1653357460,
     291743070,
     593771532,
     2941865444,
     855254640,
     2401129822,
     2420945774,
     2447532144,
     1137540092,
     1296659939,
     3252539825,
     1165427708,
     3251476781,
     2597490804,
     2518198923,
     1196242486,
     3646082981,
     1347758965,
     3824891532,
     2959519286,
     1523237529,
     2910666174,
     3226637035,
     2116458903,
     1076998092,
     4222762545,
     3061300520,
     4189298288,
     3943996060,
     3129210496,
     3826669630,
     4235952488,
     2624429853,
     2522766390,
     4137227001,
     3846448057,
     1893377487,
     3658784739,
     2368074586,
     170547540,
     520741120,
     2662229630,
     4265731754,
     1379762094,
     3395502906,
     2242123335,
     1960965916,
     561815223,
     2687853297,
     4051050259,
     1845906614,
     3725623071,
     1857706909,
     2487006596,
     1925919247,
     2796536825,
     3499954730,
     2173320675,
     3416676849,
     3637473517,
     340951464,
     4152841543,
     3747544606,
     2659955417,
     1695145107,
     3117280269,
     826143012,
     3867179892,
     4269349771,
     1002613766,
     3842086144,
     1431990957,
     2466205499,
     653575141,
     293530756,
     2318035308,
     3728576309,
     1697894989,
     2955143882,
     2109912287,
     2764187839,
     1805490664,
     672567480,
     1374741155,
     1662665091,
     3551530257,
     350283994,
     685023916,
     1887748803,
     1386316091,
     185708823,
     3106823178,
     3014109065,
     3823816879,
     2213358313,
     2696977340,
     4075569311,
     365089277,
     3466850767,
     312392153,
     1065191758,
     2405243644,
     3174745999,
     3617861250,
     867192904,
     1046475095,
     1888985494,
     1127140157,
     61671281,
     128055546,
     2332619657,
     993669439,
     2145370329,
     1462433204,
     74990676,
     2898191247,
     3601586977,
     794604597,
     3597643629,
     4282141339,
     251591051,
     84943504,
     2016044077,
     946823499,
     648214756,
     2530104367,
     4254219656,
     1974542801,
     53097687,
     157109688,
     299310673,
     2866882336,
     3335682769,
     2583612755,
     4114730718,
     740387484,
     986157357,
     1140355266,
     2825639379,
     1198731547,
     1521261313,
     1204836445,
     4294274455,
     2215732661,
     1369520150,
     1515223958,
     2428295267,
     1945985266,
     2168529560,
     3791933294,
     4021389338,
     713695045,
     4254483898,
     3795986293,
     1347498014,
     1746051095,
     1364967734,
     206265390,
     3940088473,
     1867270033,
     3893545471,
     3545819698,
     2573105187,
     3859595967,
     2823745089,
     1293424244,
     3948799370,
     1524394803,
     3807487752,
     4055830971,
     3124609223,
     119357574,
     1490516894,
     3799908122,
     1700941394,
     80878888,
     2719184407,
     3603450215,
     27225525,
     1413638246,
     3350206268,
     2643568519,
     801305037,
     1341902999,
     1420459209,
     968648411,
     1826125841,
     2619721007,
     537879916,
     860253620,
     586683700,
     827412286,
     2724526294,
     1019678576,
     3998975225,
     339789397,
     863181640,
     970475690,
     2737385140,
     322021174,
     4084948327,
     80691950,
     1702782677,
     1266230197,
     1100861683,
     3123418948,
     258978579,
     3217833394,
     1780903315,
     1345341356,
     2927579299,
     931392918,
     9404798,
     83278219,
     2470714323,
     640357359,
     2169696414,
     496463525,
     4127940882,
     2965369765,
     4136333330,
     1159134689,
     1798163043,
     4097403856,
     4284804850,
     3165524545,
     2765224926,
     931350022,
     1171636623,
     845799406,
     709853915,
     2348457302,
     3343956878,
     2438786363,
     175730452,
     598587430,
     2744955366,
     447049527,
     1252796590,
     3044128900,
     812683575,
     3721040746,
     3404688504,
     2674021068,
     959056069,
     322162714,
     2008064015,
     3758321185,
     2877937989,
     778007512,
     3502772435,
     3084124565,
     111844966,
     248248909,
     22147113,
     2506501875,
     1430033847,
     1690841637,
     2999017281,
     3658748205,
     1632773934,
     4177069459,
     3187781304,
     1182255965,
     4121685939,
     300554973,
     2854502901,
     642657206,
     1504346771,
     128405037,
     2163092164,
     1091806675,
     1144089805,
     54479906,
     505543118,
     2844153548,
     1010229282,
     2961721580,
     4235612700,
     3508832243,
     1409461040,
     2568735295,
     1191284023,
     2220949766,
     2605559386,
     706551146,
     3452279268,
     2372892169,
     2360210709,
     3228881405,
     2987444766,
     1187314024,
     908783041,
     144096950,
     1915948100,
     2171208878,
     420772043,
     793209353,
     359527746,
     625018196,
     1195796799,
     2079388581,
     864869238,
     765565143,
     1069647859,
     3857355469,
     2436437044,
     238157644,
     1612883577,
     1911189891,
     2070273440,
     384222456,
     1186369477,
     2844794758,
     3435869876,
     1486894286,
     4062343990,
     440437688,
     306253241,
     3650751868,
     2695961920,
     3920128930,
     3921419250,
     502951143,
     311093469,
     2708936678,
     36677206,
     3473343884,
     577655290,
     3795127787,
     1448118037,
     436359554,
     2051970204,
     2644913053,
     2492587228,
     3125803824,
     150160619,
     1725373463,
     2221292372,
     2580064663,
     1330289179,
     2700556441,
     1327212925,
     651999045,
     2089310372,
     3221246949,
     4148251434,
     4267892623,
     897583443,
     1051813251,
     2131903377,
     4121163297,
     4128279241,
     1634689556,
     3369895626,
     1121895497,
     3158192590,
     4290462018,
     3447288838,
     4035505534,
     2945114940,
     1556028368,
     4235061319,
     1535570089,
     2144940257,
     1961364931,
     2509075082,
     804411045,
     2290609740,
     1076471626,
     3254493188,
     4284011230,
     923006875,
     3722016670,
     2981439178,
     2038308778,
     1755166344,
     488581856,
     2624361425,
     1298790575,
     3550671725,
     1845109437,
     2047411775,
     2488464246,
     1391825885,
     2340290304,
     3623879917,
     217171099,
     3698905333,
     2718846041,
     73731529,
     2053405441,
     2770197347,
     2983996080,
     2612966141,
     2187183079,
     2796212469,
     3797629169,
     1788932364,
     17748377,
     627297271,
     3689459731,
     3311799950,
     4263162298,
     4016852324,
     3136750215,
     1725824049,
     2844064064,
     2059159211,
     3182127070,
     470655679,
     1166949584,
     2425843062,
     219908183,
     161770982,
     2394961157,
     999226372,
     2367624166,
     76287885,
     1110832227,
     3358123709,
     1504127646,
     49596774,
     1296560019,
     2320978173,
     1163934122,
     1631947491,
     2702852639,
     3856755518,
     2562943123,
     991330989,
     993726248,
     2133737192,
     20974150,
     3808389889,
     2447868340,
     2434828629,
     3344419509,
     4076789444,
     1446054487,
     3815933708,
     3644670988,
     3175898122,
     3057844745,
     559106380,
     1840065631,
     3020573012,
     3203040371,
     997381925,
     2563312032,
     815510593,
     121805231,
     1047507862,
     1841403695,
     1563170561,
     1644198099,
     3470882735,
     627296501,
     3006157508,
     383648566,
     3136652449,
     2252034149,
     1749861990,
     956381402,
     3299624735,
     2798395931,
     270054444,
     3757564211,
     2933717597,
     1080178310,
     1367392714,
     1135266342,
     2642448461,
     1067554284,
     3694982777,
     3594374699,
     4170301369,
     3593401570,
     2298071009,
     1561680798,
     2788490866,
     1757829499,
     8819607,
     2453686068,
     3458682663,
     1614888171,
     2327536307,
     13960177,
     125752716,
     2312371195,
     1515197240,
     189747227,
     666988376,
     1401118738,
     986465965,
     242793663,
     1830586663,
     1603054176,
     391536104,
     1403125754,
     4021998614,
     157985039,
     966292223,
     2476444819,
     3261614719,
     3888752449,
     2300656903,
     1138839559,
     1227396086,
     1029493665,
     2138482384,
     2182525175,
     1437393012,
     2758514342,
     1394715363,
     242430786,
     4026759135,
     379455166,
     3454852592,
     1128257576,
     513994046,
     2437643547,
     1851772774,
     1096918785,
     2537378072,
     2020382559,
     1306056753,
     519939769,
     2477462755,
     2962076712,
     2856059355,
     111272034,
     2363778749,
     3031510224,
     297098997,
     2716928589,
     1988398361,
     3715685207,
     1158387390,
     3239718824,
     214276640,
     1240159361,
     302800084,
     258391670,
     3118615408,
     1789752935,
     935790045,
     1678444383,
     3645357112,
     1752731774,
     1211889371,
     2432949496,
     1983838022,
     2563701701,
     3235972690,
     2732559614,
     4173627589,
     918129740,
     3528101943,
     945287787,
     783593046,
     1687101911,
     4265659819,
     1625936204,
     419423123,
     404748783,
     174814826,
     561306387,
     441376876,
     3649973873,
     1191532754,
     493829681,
     462640703,
     3037639795,
     4234288143,
     787992128,
     354556603,
     1391557094,
     1227150157,
     25592400,
     3032298621,
     1655829692,
     1736544192,
     2936173068,
     1867683432,
     3284761215,
     2988749127,
     62083315,
     3675433852,
     1134152479,
     2537382040,
     1147996351,
     1287284159,
     1889610942,
     3549411223,
     2634772335,
     1621708033,
     3268420142,
     2635222095,
     2856377255,
     3703296204,
     45831019,
     1997278369,
     1472530726,
     4202051236,
     1958581642,
     1899513707,
     1642075765,
     217373156,
     1177071505,
     2179831909,
     1894821896,
     375785474,
     140181353,
     2743987480,
     123627609,
     3644816362,
     4244769687,
     4053481902,
     4272740073,
     1701735471,
     1799303028,
     2810175160,
     1531107068,
     3059813822,
     4125025775,
     1932301928,
     358163550,
     1246286294,
     1901878857,
     2449370117,
     4061706076,
     2875797072,
     1661522553,
     543545982,
     300448222,
     4019581644,
     3197346443,
     731278538,
     457112622,
     669625172,
     2548620393,
     2931934447,
     2318225955,
     427149964,
     1097556601,
     3585697077,
     1901391738,
     3019912350,
     4193989774,
     1411691495,
     2549773310,
     3130489018,
     739444137,
     1953561922,
     228589899,
     974825144,
     1873934953,
     918502475,
     4020302125,
     2103082289,
     1474428456,
     269315616,
     3376419786,
     2903506696,
     169344159,
     4151327830,
     2861975985,
     1583628545,
     337656074,
     2381206238,
     1346357469,
     3316549550,
     1188140897,
     928463634,
     120466083,
     1048016215,
     2053770646,
     3729204448,
     3630812747,
     3421817962,
     1471357089,
     2971633393,
     2721366758,
     3977792328,
     2771228423,
     258029855,
     325097628,
     2816869331,
     228010778,
     1815596248,
     2677647806,
     4069826588,
     2009464559,
     4003870353,
     2558198381,
     823508134,
     256895388,
     130455482,
     4107398577,
     2446165146,
     3086759840,
     3128842794,
     236454548,
     3740649072,
     1049081391,
     3780795812,
     1964380357,
     3900635454,
     1941196066,
     1143285596,
     1276856333,
     2919547816,
     2947639569,
     1889305089,
     2386910172,
     2685680362,
     2042792556,
     2780968041,
     976912013,
     3562274424,
     2336140155,
     3464857244,
     1108365812,
     1201566469,
     707126700,
     4047776595,
     1289380202,
     1231913128,
     2819729319,
     537908270,
     3802355886,
     2004615093,
     2947614997,
     4192189156,
     2809733754,
     3082820238,
     2758499499,
     1004612882,
     1102702383,
     1862546275,
     3170345990,
     883739952,
     1641198615,
     957782688,
     1503652889,
     2210400768,
     2002162781,
     1553086024,
     2591721606,
     3830165160,
     4181044959,
     2735782270,
     3825677158,
     143739895,
     771193452,
     35990560,
     1014009970,
     20768744,
     1785268932,
     1424740580,
     1620237280,
     848157259,
     3808893671,
     2746756110,
     3903639825,
     1822084165,
     2891666588,
     3853186896,
     4248495212,
     1178592425,
     455721495,
     1848821934,
     1558397701,
     133397899,
     1845531767,
     2798312897,
     1471176399,
     1743248506,
     2229972777,
     1290369879,
     3579075953,
     309034994,
     929728690,
     3841454719,
     3031753515,
     3606461413,
     2412281758,
     2993123515,
   };
   int const nrandoms = sizeof(randoms) / sizeof(*randoms);
   
   int const bits = count_bits(sgtype);
   sgtype const tmin =
     is_signed(sgtype) ? ((sgtype)1 << (sgtype)(bits-1)) : (sgtype)0;
   sgtype const tmax = tmin - (sgtype)1;
   for (int iter=0; iter<nrandoms; ++iter) {
     typedef union {
       gtype  v;
       ugtype u;
       sgtype s[16];
     } Tvec;
     Tvec x, y, z;
     Tvec good_abs;
     Tvec good_abs_diff, good_add_sat;
     Tvec good_hadd, good_rhadd;
     int vecsize = vec_step(gtype);
     for (int n=0; n<vecsize; ++n) {
       x.s[n] = randoms[(iter+n   ) % nrandoms];
       y.s[n] = randoms[(iter+n+20) % nrandoms];
       z.s[n] = randoms[(iter+n+40) % nrandoms];
       if (bits>32) {
         x.s[n] = (x.s[n] << (bits/2)) | randoms[(iter+n+100) % nrandoms];
         y.s[n] = (y.s[n] << (bits/2)) | randoms[(iter+n+120) % nrandoms];
         z.s[n] = (z.s[n] << (bits/2)) | randoms[(iter+n+140) % nrandoms];
       }
       good_abs.s[n] =
         safe_extract(safe_abs(safe_create(x.s[n])));
       good_abs_diff.s[n] =
         safe_extract(safe_abs(safe_sub(safe_create(x.s[n]),
                                        safe_create(y.s[n]))));
       good_add_sat.s[n] =
         safe_extract(safe_min(safe_max(safe_add(safe_create(x.s[n]),
                                                 safe_create(y.s[n])),
                                        safe_create(tmin)),
                               safe_create(tmax)));
       good_hadd.s[n] =
         safe_extract(safe_rshift(safe_add(safe_create(x.s[n]),
                                           safe_create(y.s[n]))));
       good_rhadd.s[n] =
         safe_extract(safe_rshift(safe_add(safe_add(safe_create(x.s[n]),
                                                    safe_create(y.s[n])),
                                           safe_create((sgtype)1))));
     }
     Tvec res_abs;
     Tvec res_abs_diff, res_add_sat;
     Tvec res_hadd, res_rhadd;
     res_abs.u      = abs     (x.v);
     res_abs_diff.u = abs_diff(x.v, y.v);
     res_add_sat.v  = add_sat (x.v, y.v);
     res_hadd.v     = hadd    (x.v, y.v);
     res_rhadd.v    = rhadd   (x.v, y.v);
     bool equal;
     // abs
     equal = true;
     for (int n=0; n<vecsize; ++n) {
       equal = equal && res_abs.s[n] == good_abs.s[n];
     }
     if (!equal) {
       printf("FAIL: abs type=%s\n", typename);
       for (int n=0; n<vecsize; ++n) {
         printf("   [%d] a=%d good=%d res=%d\n",
                n,
                (int)x.s[n],
                (int)good_abs.s[n], (int)res_abs.s[n]);
       }
       return;
     }
     // abs_diff
     equal = true;
     for (int n=0; n<vecsize; ++n) {
       equal = equal && res_abs_diff.s[n] == good_abs_diff.s[n];
     }
     if (!equal) {
       printf("FAIL: abs_diff type=%s\n", typename);
       for (int n=0; n<vecsize; ++n) {
         printf("   [%d] a=%d b=%d good=%d res=%d\n",
                n,
                (int)x.s[n], (int)y.s[n],
                (int)good_abs_diff.s[n], (int)res_abs_diff.s[n]);
       }
       return;
     }
     // add_sat
     equal = true;
     for (int n=0; n<vecsize; ++n) {
       equal = equal && res_add_sat.s[n] == good_add_sat.s[n];
     }
     if (!equal) {
       printf("FAIL: add_sat type=%s\n", typename);
       for (int n=0; n<vecsize; ++n) {
         printf("   [%d] a=%d b=%d good=%d res=%d\n",
                n,
                (int)x.s[n], (int)y.s[n],
                (int)good_add_sat.s[n], (int)res_add_sat.s[n]);
       }
       return;
     }
     // hadd
     equal = true;
     for (int n=0; n<vecsize; ++n) {
       equal = equal && res_hadd.s[n] == good_hadd.s[n];
     }
     if (!equal) {
       printf("FAIL: hadd type=%s\n", typename);
       for (int n=0; n<vecsize; ++n) {
         printf("   [%d] a=%d b=%d good=%d res=%d\n",
                n,
                (int)x.s[n], (int)y.s[n],
                (int)good_hadd.s[n], (int)res_hadd.s[n]);
       }
       return;
     }
     // rhadd
     equal = true;
     for (int n=0; n<vecsize; ++n) {
       equal = equal && res_rhadd.s[n] == good_rhadd.s[n];
     }
     if (!equal) {
       printf("FAIL: rhadd type=%s\n", typename);
       for (int n=0; n<vecsize; ++n) {
         printf("   [%d] a=%d b=%d good=%d res=%d\n",
                n,
                (int)x.s[n], (int)y.s[n],
                (int)good_rhadd.s[n], (int)res_rhadd.s[n]);
       }
       return;
     }
   }
 })
 )

kernel void test_hadd()
{
  CALL_FUNC_G(test_hadd)
}
