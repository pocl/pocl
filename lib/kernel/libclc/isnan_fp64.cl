#ifdef SINGLEVEC
_CL_OVERLOADABLE inttype isnan(vtype i)
#else
_CL_OVERLOADABLE itype isnan(vtype i)
#endif
{
  itype res = ((as_utype(i) << 1) > (utype)((ulong)EXPBITS_DP64 << 1));
#ifdef SINGLEVEC
  return convert_int(res);
#else
  return res;
#endif
}
