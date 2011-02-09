__kernel void 
dot_product (__global const float4 *a,  
	     __global const float4 *b, __global float *c) 
{ 
  int gid = get_global_id(0); 

  c[gid] = dot(a[gid], b[gid]); 
}
