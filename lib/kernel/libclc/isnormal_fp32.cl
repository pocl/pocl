_CL_OVERLOADABLE itype isnormal(vtype i)
{
  return ((as_utype(i) << 1) < (utype)(EXPBITS_SP32 << 1)) & ((as_utype(i) << 1) > (utype)(MANTBITS_SP32 << 1));
}
