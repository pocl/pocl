_CL_OVERLOADABLE itype isinf(vtype i)
{
  return ((as_itype(i) << 1) == (itype)(EXPBITS_SP32 << 1));
}
