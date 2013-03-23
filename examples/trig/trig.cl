__kernel void
trig(__global const float4 *a, __global float4 *b, float c)
{
  int gid = get_global_id(0);
  
  switch (gid % 5) {
  case 0: b[gid] = cos(a[gid]); break;
  case 1: b[gid] = fabs(a[gid])+c; break;
  case 2: b[gid] = sin(a[gid]); break;
  case 3: b[gid] = sqrt(a[gid]); break;
  case 4: b[gid] = tan(a[gid]); break;
  }
}
