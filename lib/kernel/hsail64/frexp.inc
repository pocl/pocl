float _CL_OVERLOADABLE frexp(float x, ADDRSPACE int* irp)
{
  int e = LSR_F(x);
  int ir = e - (int)(PROPS_FLOAT_EXPONENT_OFFSET - 1);
  //ir = ifthen(convert_bool(e), ir, IV(std::numeric_limits<int_t>::min()));
  if (!(e))
    ir = INT_MIN;
#if defined VML_HAVE_INF
  //ir = ifthen(isinf(x), IV(std::numeric_limits<int_t>::max()), ir);
  if (isinf(x))
    ir = INT_MAX;
#endif
#if defined VML_HAVE_NAN
  //ir = ifthen(isnan(x), IV(std::numeric_limits<int_t>::min()), ir);
  if (isnan(x))
    ir = INT_MIN;
#endif
  float r =
    as_float((as_int(x) & (int)(PROPS_FLOAT_SIGNBIT_MASK | PROPS_FLOAT_MANTISSA_MASK)) |
             (int)(as_int(0.5f) & PROPS_FLOAT_EXPONENT_MASK));
  // bool iszero = (x == 0.0f);
  if (x == 0.0f)
    {
      //ir = ifthen(iszero, IV(I(0)), ir);
      //r = ifthen(iszero, copysign(RV(R(0.0)), r), r);
      ir = 0;
      r = copysign(0.0f, r);
    }
  *irp = ir;
  return r;
}

double _CL_OVERLOADABLE frexp(double x, ADDRSPACE int* irp)
{
  int e = LSR_D(x);
  int ir = e - (int)(PROPS_DOUBLE_EXPONENT_OFFSET - 1);
  //ir = ifthen(convert_bool(e), ir, IV(std::numeric_limits<long_t>::min()));
  if (!(e))
    ir = INT_MIN;
#if defined VML_HAVE_INF
  //ir = ifthen(isinf(x), IV(std::numeric_limits<long_t>::max()), ir);
  if (isinf(x))
    ir = INT_MAX;
#endif
#if defined VML_HAVE_NAN
  //ir = ifthen(isnan(x), IV(std::numeric_limits<long_t>::min()), ir);
  if (isnan(x))
    ir = INT_MIN;
#endif
  double r =
    as_double((as_long(x) & (long)(PROPS_DOUBLE_SIGNBIT_MASK | PROPS_DOUBLE_MANTISSA_MASK)) |
             (as_long(0.5f) & PROPS_DOUBLE_EXPONENT_MASK));
  // bool iszero = (x == 0.0f);
  if (x == 0.0)
    {
      //ir = ifthen(iszero, IV(I(0)), ir);
      //r = ifthen(iszero, copysign(RV(R(0.0)), r), r);
      ir = 0;
      r = copysign(0.0, r);
    }
  *irp = ir;
  return r;
}
