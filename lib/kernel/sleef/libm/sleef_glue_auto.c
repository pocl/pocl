#include "sleef.h"

#include "sleef_cl.h"


#ifdef SLEEF_VEC_128_AVAILABLE

_CL_ALWAYSINLINE float4 Sleef_sinf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_sinf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_sinf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_sinf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_sind2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_sind2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_sind2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_sind2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_cosf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_cosf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_cosf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_cosf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_cosd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_cosd2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_cosd2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_cosd2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_tanf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_tanf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_tanf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_tanf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_tand2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_tand2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_tand2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_tand2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_asinf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_asinf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_asinf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_asinf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_asind2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_asind2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_asind2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_asind2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_acosf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_acosf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_acosf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_acosf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_acosd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_acosd2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_acosd2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_acosd2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_atanf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_atanf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_atanf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_atanf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_atand2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_atand2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_atand2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_atand2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_atan2f4_u10(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_atan2f4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_atan2f4_u35(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_atan2f4_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_atan2d2_u10(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_atan2d2_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_atan2d2_u35(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_atan2d2_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_cbrtf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_cbrtf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_cbrtf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_cbrtf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_cbrtd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_cbrtd2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_cbrtd2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_cbrtd2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_logf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_logf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_logf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_logf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_logd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_logd2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_logd2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_logd2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_expf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_expf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_expd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_expd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_powf4_u10(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_powf4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_powd2_u10(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_powd2_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_pownf4_u10(float4 x, int4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_pownf4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_pownd2_u10_long(double2 x, long2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { long2 t; reg128i r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_pownd2_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_powrf4_u10(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_powrf4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_powrd2_u10(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_powrd2_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_sinhf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_sinhf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_sinhd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_sinhd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_coshf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_coshf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_coshd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_coshd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_tanhf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_tanhf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_tanhd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_tanhd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_asinhf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_asinhf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_asinhd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_asinhd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_acoshf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_acoshf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_acoshd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_acoshd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_atanhf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_atanhf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_atanhd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_atanhd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_exp2f4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_exp2f4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_exp2d2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_exp2d2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_exp10f4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_exp10f4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_exp10d2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_exp10d2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_expm1f4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_expm1f4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_expm1d2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_expm1d2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_log10f4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_log10f4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_log10d2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_log10d2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_log1pf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_log1pf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_log1pd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_log1pd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_sinpif4_u05(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_sinpif4_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_sinpid2_u05(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_sinpid2_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_cospif4_u05(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_cospif4_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_cospid2_u05(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_cospid2_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_fmaf4(float4 x, float4 y, float4 z)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } z_in;
  z_in.t = z;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_fmaf4_intrin(x_in.r, y_in.r, z_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_fmad2(double2 x, double2 y, double2 z)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } z_in;
  z_in.t = z;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_fmad2_intrin(x_in.r, y_in.r, z_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_sqrtf4_u05(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_sqrtf4_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_sqrtd2_u05(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_sqrtd2_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_hypotf4_u05(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_hypotf4_u05_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float4 Sleef_hypotf4_u35(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_hypotf4_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_hypotd2_u05(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_hypotd2_u05_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double2 Sleef_hypotd2_u35(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_hypotd2_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_fabsf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_fabsf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_fabsd2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_fabsd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_copysignf4(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_copysignf4_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_copysignd2(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_copysignd2_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_fmaxf4(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_fmaxf4_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_fmaxd2(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_fmaxd2_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_fminf4(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_fminf4_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_fmind2(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_fmind2_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_fdimf4(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_fdimf4_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_fdimd2(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_fdimd2_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_truncf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_truncf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_truncd2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_truncd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_floorf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_floorf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_floord2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_floord2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_ceilf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_ceilf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_ceild2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_ceild2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_roundf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_roundf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_roundd2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_roundd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_rintf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_rintf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_rintd2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_rintd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_nextafterf4(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_nextafterf4_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_nextafterd2(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_nextafterd2_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_fmodf4(float4 x, float4 y)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } y_in;
  y_in.t = y;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_fmodf4_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_fmodd2(double2 x, double2 y)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } y_in;
  y_in.t = y;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_fmodd2_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_lgammaf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_lgammaf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_lgammad2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_lgammad2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_tgammaf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_tgammaf4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_tgammad2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_tgammad2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_erff4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_erff4_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_erfd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_erfd2_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_erfcf4_u15(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_erfcf4_u15_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_erfcd2_u15(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_erfcd2_u15_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_frfrexpf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_frfrexpf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_frfrexpd2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_frfrexpd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float4_2 Sleef_sincosf4_u10(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { Sleef_float4_2 t; Sleef_reg128f_2 r; } ret;
  ret.r = Sleef_sincosf4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_float4_2 Sleef_sincosf4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { Sleef_float4_2 t; Sleef_reg128f_2 r; } ret;
  ret.r = Sleef_sincosf4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double2_2 Sleef_sincosd2_u10(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { Sleef_double2_2 t; Sleef_reg128d_2 r; } ret;
  ret.r = Sleef_sincosd2_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_double2_2 Sleef_sincosd2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { Sleef_double2_2 t; Sleef_reg128d_2 r; } ret;
  ret.r = Sleef_sincosd2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float4_2 Sleef_sincospif4_u05(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { Sleef_float4_2 t; Sleef_reg128f_2 r; } ret;
  ret.r = Sleef_sincospif4_u05_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_float4_2 Sleef_sincospif4_u35(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { Sleef_float4_2 t; Sleef_reg128f_2 r; } ret;
  ret.r = Sleef_sincospif4_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double2_2 Sleef_sincospid2_u05(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { Sleef_double2_2 t; Sleef_reg128d_2 r; } ret;
  ret.r = Sleef_sincospid2_u05_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_double2_2 Sleef_sincospid2_u35(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { Sleef_double2_2 t; Sleef_reg128d_2 r; } ret;
  ret.r = Sleef_sincospid2_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float4_2 Sleef_modff4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { Sleef_float4_2 t; Sleef_reg128f_2 r; } ret;
  ret.r = Sleef_modff4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double2_2 Sleef_modfd2(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { Sleef_double2_2 t; Sleef_reg128d_2 r; } ret;
  ret.r = Sleef_modfd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float4 Sleef_ldexpf4(float4 x, int4 k)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } k_in;
  k_in.t = k;
  union { float4 t; reg128f r; } ret;
  ret.r = Sleef_ldexpf4_intrin(x_in.r, k_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_ldexpd2_long(double2 x, long2 k)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { long2 t; reg128i r; } k_in;
  k_in.t = k;
  union { double2 t; reg128d r; } ret;
  ret.r = Sleef_ldexpd2_intrin(x_in.r, k_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE int4 Sleef_expfrexpf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } ret;
  ret.r = Sleef_expfrexpf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE long2 Sleef_expfrexpd2_long(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { long2 t; reg128i r; } ret;
  ret.r = Sleef_expfrexpd2_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE int4 Sleef_ilogbf4(float4 x)
{
  union { float4 t; reg128f r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } ret;
  ret.r = Sleef_ilogbf4_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE long2 Sleef_ilogbd2_long(double2 x)
{
  union { double2 t; reg128d r; } x_in;
  x_in.t = x;
  union { long2 t; reg128i r; } ret;
  ret.r = Sleef_ilogbd2_intrin(x_in.r);
  return ret.t;
}
#endif

#endif


#ifdef SLEEF_VEC_256_AVAILABLE

_CL_ALWAYSINLINE float8 Sleef_sinf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_sinf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_sinf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_sinf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_sind4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_sind4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_sind4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_sind4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_cosf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_cosf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_cosf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_cosf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_cosd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_cosd4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_cosd4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_cosd4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_tanf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_tanf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_tanf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_tanf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_tand4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_tand4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_tand4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_tand4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_asinf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_asinf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_asinf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_asinf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_asind4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_asind4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_asind4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_asind4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_acosf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_acosf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_acosf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_acosf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_acosd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_acosd4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_acosd4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_acosd4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_atanf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_atanf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_atanf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_atanf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_atand4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_atand4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_atand4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_atand4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_atan2f8_u10(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_atan2f8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_atan2f8_u35(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_atan2f8_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_atan2d4_u10(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_atan2d4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_atan2d4_u35(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_atan2d4_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_cbrtf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_cbrtf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_cbrtf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_cbrtf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_cbrtd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_cbrtd4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_cbrtd4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_cbrtd4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_logf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_logf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_logf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_logf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_logd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_logd4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_logd4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_logd4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_expf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_expf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_expd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_expd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_powf8_u10(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_powf8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_powd4_u10(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_powd4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_pownf8_u10(float8 x, int8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_pownf8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_pownd4_u10(double4 x, int4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_pownd4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_powrf8_u10(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_powrf8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_powrd4_u10(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_powrd4_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_sinhf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_sinhf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_sinhd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_sinhd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_coshf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_coshf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_coshd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_coshd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_tanhf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_tanhf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_tanhd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_tanhd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_asinhf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_asinhf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_asinhd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_asinhd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_acoshf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_acoshf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_acoshd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_acoshd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_atanhf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_atanhf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_atanhd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_atanhd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_exp2f8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_exp2f8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_exp2d4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_exp2d4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_exp10f8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_exp10f8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_exp10d4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_exp10d4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_expm1f8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_expm1f8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_expm1d4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_expm1d4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_log10f8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_log10f8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_log10d4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_log10d4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_log1pf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_log1pf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_log1pd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_log1pd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_sinpif8_u05(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_sinpif8_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_sinpid4_u05(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_sinpid4_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_cospif8_u05(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_cospif8_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_cospid4_u05(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_cospid4_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_fmaf8(float8 x, float8 y, float8 z)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } z_in;
  z_in.t = z;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_fmaf8_intrin(x_in.r, y_in.r, z_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_fmad4(double4 x, double4 y, double4 z)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } z_in;
  z_in.t = z;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_fmad4_intrin(x_in.r, y_in.r, z_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_sqrtf8_u05(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_sqrtf8_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_sqrtd4_u05(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_sqrtd4_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_hypotf8_u05(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_hypotf8_u05_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float8 Sleef_hypotf8_u35(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_hypotf8_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_hypotd4_u05(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_hypotd4_u05_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double4 Sleef_hypotd4_u35(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_hypotd4_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_fabsf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_fabsf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_fabsd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_fabsd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_copysignf8(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_copysignf8_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_copysignd4(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_copysignd4_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_fmaxf8(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_fmaxf8_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_fmaxd4(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_fmaxd4_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_fminf8(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_fminf8_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_fmind4(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_fmind4_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_fdimf8(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_fdimf8_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_fdimd4(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_fdimd4_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_truncf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_truncf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_truncd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_truncd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_floorf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_floorf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_floord4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_floord4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_ceilf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_ceilf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_ceild4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_ceild4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_roundf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_roundf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_roundd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_roundd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_rintf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_rintf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_rintd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_rintd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_nextafterf8(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_nextafterf8_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_nextafterd4(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_nextafterd4_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_fmodf8(float8 x, float8 y)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } y_in;
  y_in.t = y;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_fmodf8_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_fmodd4(double4 x, double4 y)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } y_in;
  y_in.t = y;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_fmodd4_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_lgammaf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_lgammaf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_lgammad4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_lgammad4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_tgammaf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_tgammaf8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_tgammad4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_tgammad4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_erff8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_erff8_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_erfd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_erfd4_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_erfcf8_u15(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_erfcf8_u15_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_erfcd4_u15(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_erfcd4_u15_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_frfrexpf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_frfrexpf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_frfrexpd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_frfrexpd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float8_2 Sleef_sincosf8_u10(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { Sleef_float8_2 t; Sleef_reg256f_2 r; } ret;
  ret.r = Sleef_sincosf8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_float8_2 Sleef_sincosf8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { Sleef_float8_2 t; Sleef_reg256f_2 r; } ret;
  ret.r = Sleef_sincosf8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double4_2 Sleef_sincosd4_u10(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { Sleef_double4_2 t; Sleef_reg256d_2 r; } ret;
  ret.r = Sleef_sincosd4_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_double4_2 Sleef_sincosd4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { Sleef_double4_2 t; Sleef_reg256d_2 r; } ret;
  ret.r = Sleef_sincosd4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float8_2 Sleef_sincospif8_u05(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { Sleef_float8_2 t; Sleef_reg256f_2 r; } ret;
  ret.r = Sleef_sincospif8_u05_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_float8_2 Sleef_sincospif8_u35(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { Sleef_float8_2 t; Sleef_reg256f_2 r; } ret;
  ret.r = Sleef_sincospif8_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double4_2 Sleef_sincospid4_u05(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { Sleef_double4_2 t; Sleef_reg256d_2 r; } ret;
  ret.r = Sleef_sincospid4_u05_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_double4_2 Sleef_sincospid4_u35(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { Sleef_double4_2 t; Sleef_reg256d_2 r; } ret;
  ret.r = Sleef_sincospid4_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float8_2 Sleef_modff8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { Sleef_float8_2 t; Sleef_reg256f_2 r; } ret;
  ret.r = Sleef_modff8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double4_2 Sleef_modfd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { Sleef_double4_2 t; Sleef_reg256d_2 r; } ret;
  ret.r = Sleef_modfd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float8 Sleef_ldexpf8(float8 x, int8 k)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } k_in;
  k_in.t = k;
  union { float8 t; reg256f r; } ret;
  ret.r = Sleef_ldexpf8_intrin(x_in.r, k_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double4 Sleef_ldexpd4(double4 x, int4 k)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } k_in;
  k_in.t = k;
  union { double4 t; reg256d r; } ret;
  ret.r = Sleef_ldexpd4_intrin(x_in.r, k_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE int8 Sleef_expfrexpf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } ret;
  ret.r = Sleef_expfrexpf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE long4 Sleef_expfrexpd4_long(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { long4 t; reg256i r; } ret;
  ret.r = Sleef_expfrexpd4_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE int8 Sleef_ilogbf8(float8 x)
{
  union { float8 t; reg256f r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } ret;
  ret.r = Sleef_ilogbf8_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE int4 Sleef_ilogbd4(double4 x)
{
  union { double4 t; reg256d r; } x_in;
  x_in.t = x;
  union { int4 t; reg128i r; } ret;
  ret.r = Sleef_ilogbd4_intrin(x_in.r);
  return ret.t;
}
#endif

#endif


#ifdef SLEEF_VEC_512_AVAILABLE

_CL_ALWAYSINLINE float16 Sleef_sinf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_sinf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_sinf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_sinf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_sind8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_sind8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_sind8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_sind8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_cosf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_cosf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_cosf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_cosf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_cosd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_cosd8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_cosd8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_cosd8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_tanf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_tanf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_tanf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_tanf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_tand8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_tand8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_tand8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_tand8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_asinf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_asinf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_asinf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_asinf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_asind8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_asind8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_asind8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_asind8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_acosf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_acosf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_acosf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_acosf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_acosd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_acosd8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_acosd8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_acosd8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_atanf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_atanf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_atanf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_atanf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_atand8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_atand8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_atand8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_atand8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_atan2f16_u10(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_atan2f16_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_atan2f16_u35(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_atan2f16_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_atan2d8_u10(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_atan2d8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_atan2d8_u35(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_atan2d8_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_cbrtf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_cbrtf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_cbrtf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_cbrtf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_cbrtd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_cbrtd8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_cbrtd8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_cbrtd8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_logf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_logf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_logf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_logf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_logd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_logd8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_logd8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_logd8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_expf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_expf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_expd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_expd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_powf16_u10(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_powf16_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_powd8_u10(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_powd8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_pownf16_u10(float16 x, int16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { int16 t; reg512i r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_pownf16_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_pownd8_u10(double8 x, int8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_pownd8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_powrf16_u10(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_powrf16_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_powrd8_u10(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_powrd8_u10_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_sinhf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_sinhf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_sinhd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_sinhd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_coshf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_coshf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_coshd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_coshd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_tanhf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_tanhf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_tanhd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_tanhd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_asinhf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_asinhf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_asinhd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_asinhd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_acoshf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_acoshf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_acoshd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_acoshd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_atanhf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_atanhf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_atanhd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_atanhd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_exp2f16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_exp2f16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_exp2d8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_exp2d8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_exp10f16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_exp10f16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_exp10d8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_exp10d8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_expm1f16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_expm1f16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_expm1d8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_expm1d8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_log10f16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_log10f16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_log10d8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_log10d8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_log1pf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_log1pf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_log1pd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_log1pd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_sinpif16_u05(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_sinpif16_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_sinpid8_u05(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_sinpid8_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_cospif16_u05(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_cospif16_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_cospid8_u05(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_cospid8_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_fmaf16(float16 x, float16 y, float16 z)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } z_in;
  z_in.t = z;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_fmaf16_intrin(x_in.r, y_in.r, z_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_fmad8(double8 x, double8 y, double8 z)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } z_in;
  z_in.t = z;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_fmad8_intrin(x_in.r, y_in.r, z_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_sqrtf16_u05(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_sqrtf16_u05_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_sqrtd8_u05(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_sqrtd8_u05_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_hypotf16_u05(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_hypotf16_u05_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE float16 Sleef_hypotf16_u35(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_hypotf16_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_hypotd8_u05(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_hypotd8_u05_intrin(x_in.r, y_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE double8 Sleef_hypotd8_u35(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_hypotd8_u35_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_fabsf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_fabsf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_fabsd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_fabsd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_copysignf16(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_copysignf16_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_copysignd8(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_copysignd8_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_fmaxf16(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_fmaxf16_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_fmaxd8(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_fmaxd8_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_fminf16(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_fminf16_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_fmind8(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_fmind8_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_fdimf16(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_fdimf16_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_fdimd8(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_fdimd8_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_truncf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_truncf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_truncd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_truncd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_floorf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_floorf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_floord8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_floord8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_ceilf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_ceilf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_ceild8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_ceild8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_roundf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_roundf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_roundd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_roundd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_rintf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_rintf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_rintd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_rintd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_nextafterf16(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_nextafterf16_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_nextafterd8(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_nextafterd8_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_fmodf16(float16 x, float16 y)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } y_in;
  y_in.t = y;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_fmodf16_intrin(x_in.r, y_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_fmodd8(double8 x, double8 y)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } y_in;
  y_in.t = y;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_fmodd8_intrin(x_in.r, y_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_lgammaf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_lgammaf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_lgammad8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_lgammad8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_tgammaf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_tgammaf16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_tgammad8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_tgammad8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_erff16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_erff16_u10_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_erfd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_erfd8_u10_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_erfcf16_u15(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_erfcf16_u15_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_erfcd8_u15(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_erfcd8_u15_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_frfrexpf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_frfrexpf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_frfrexpd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_frfrexpd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float16_2 Sleef_sincosf16_u10(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { Sleef_float16_2 t; Sleef_reg512f_2 r; } ret;
  ret.r = Sleef_sincosf16_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_float16_2 Sleef_sincosf16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { Sleef_float16_2 t; Sleef_reg512f_2 r; } ret;
  ret.r = Sleef_sincosf16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double8_2 Sleef_sincosd8_u10(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { Sleef_double8_2 t; Sleef_reg512d_2 r; } ret;
  ret.r = Sleef_sincosd8_u10_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_double8_2 Sleef_sincosd8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { Sleef_double8_2 t; Sleef_reg512d_2 r; } ret;
  ret.r = Sleef_sincosd8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float16_2 Sleef_sincospif16_u05(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { Sleef_float16_2 t; Sleef_reg512f_2 r; } ret;
  ret.r = Sleef_sincospif16_u05_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_float16_2 Sleef_sincospif16_u35(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { Sleef_float16_2 t; Sleef_reg512f_2 r; } ret;
  ret.r = Sleef_sincospif16_u35_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double8_2 Sleef_sincospid8_u05(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { Sleef_double8_2 t; Sleef_reg512d_2 r; } ret;
  ret.r = Sleef_sincospid8_u05_intrin(x_in.r);
  return ret.t;
}

_CL_ALWAYSINLINE Sleef_double8_2 Sleef_sincospid8_u35(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { Sleef_double8_2 t; Sleef_reg512d_2 r; } ret;
  ret.r = Sleef_sincospid8_u35_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE Sleef_float16_2 Sleef_modff16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { Sleef_float16_2 t; Sleef_reg512f_2 r; } ret;
  ret.r = Sleef_modff16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE Sleef_double8_2 Sleef_modfd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { Sleef_double8_2 t; Sleef_reg512d_2 r; } ret;
  ret.r = Sleef_modfd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE float16 Sleef_ldexpf16(float16 x, int16 k)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { int16 t; reg512i r; } k_in;
  k_in.t = k;
  union { float16 t; reg512f r; } ret;
  ret.r = Sleef_ldexpf16_intrin(x_in.r, k_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE double8 Sleef_ldexpd8(double8 x, int8 k)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } k_in;
  k_in.t = k;
  union { double8 t; reg512d r; } ret;
  ret.r = Sleef_ldexpd8_intrin(x_in.r, k_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE int16 Sleef_expfrexpf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { int16 t; reg512i r; } ret;
  ret.r = Sleef_expfrexpf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE long8 Sleef_expfrexpd8_long(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { long8 t; reg512i r; } ret;
  ret.r = Sleef_expfrexpd8_intrin(x_in.r);
  return ret.t;
}
#endif


_CL_ALWAYSINLINE int16 Sleef_ilogbf16(float16 x)
{
  union { float16 t; reg512f r; } x_in;
  x_in.t = x;
  union { int16 t; reg512i r; } ret;
  ret.r = Sleef_ilogbf16_intrin(x_in.r);
  return ret.t;
}

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE

_CL_ALWAYSINLINE int8 Sleef_ilogbd8(double8 x)
{
  union { double8 t; reg512d r; } x_in;
  x_in.t = x;
  union { int8 t; reg256i r; } ret;
  ret.r = Sleef_ilogbd8_intrin(x_in.r);
  return ret.t;
}
#endif

#endif
