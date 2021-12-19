
_CL_OVERLOADABLE vtype nextafter(vtype x, vtype y) {

  const itype sign_bit
   = (itype)1 << (sizeof(itype) * 8 - 1);
  const itype sign_bit_mask = as_itype(as_utype(sign_bit) - (utype)1);

  itype ix = as_itype(x);
  itype ax = ix & sign_bit_mask;
  itype mx = sign_bit - ix;
  mx = ix < (itype)0 ? mx : ix;
  itype iy = as_itype(y);
  itype ay = iy & sign_bit_mask;
  itype my = sign_bit - iy;
  my = iy < (itype)0 ? my : iy;
  itype t = mx + (mx < my ? 1 : -1);
  itype r = sign_bit - t;
  r = t < (itype)0 ? r : t;
  r = isnan(x) ? ix : r;
  r = isnan(y) ? iy : r;
  r = ((ax | ay) == (itype)0 | ix == iy) ? iy : r;
  return as_vtype(r);
}
