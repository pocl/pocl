typedef struct {
  int c00;
  int c01;
} scan_t;

scan_t add(scan_t a, scan_t b)
{
  return b;
}

kernel
void test_local_struct_array()
{
  local scan_t psc_ldata[2];

  psc_ldata[0] = add(psc_ldata[0], psc_ldata[1]);
}
