#ifdef SINGLEVEC
_CL_OVERLOADABLE inttype isinf(vtype i)
#else
_CL_OVERLOADABLE itype isinf(vtype i)
#endif
{
  itype res = ((as_itype(i) << 1) == (itype)(EXPBITS_DP64 << 1));
#ifdef SINGLEVEC
  return convert_int(res);
#else
  return res;
#endif
}
