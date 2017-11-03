#ifdef SINGLEVEC
_CL_OVERLOADABLE inttype isfinite(vtype i)
#else
_CL_OVERLOADABLE itype isfinite(vtype i)
#endif
{
  itype res = ((as_utype(i) << 1) < (utype)(EXPBITS_DP64 << 1));
#ifdef SINGLEVEC
  return convert_int(res);
#else
  return res;
#endif
}
