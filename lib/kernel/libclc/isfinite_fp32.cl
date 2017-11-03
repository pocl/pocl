_CL_OVERLOADABLE itype isfinite(vtype i)
{
  return ((as_utype(i) << 1) < (utype)(EXPBITS_SP32 << 1));
}
