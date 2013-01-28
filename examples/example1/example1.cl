__kernel void 
dot_product (__global const float4 *a,  
	     __global const float4 *b, __global float *c) 
{ 
  int gid = get_global_id(0); 
  
  barrier(CLK_LOCAL_MEM_FENCE);

  c[gid] = a[gid].x * b[gid].x;
  c[gid] += a[gid].y * b[gid].y;
  c[gid] += a[gid].z * b[gid].z;
  c[gid] += a[gid].w * b[gid].w;
}
