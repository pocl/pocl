__attribute__((reqd_work_group_size(2, 2, 2)))
__kernel void
dot_product222 (__global const float4 *a,  
	        __global const float4 *b, __global float *c) 
{ 
  int gid = get_global_id(0); 

  c[gid] = dot(a[gid], b[gid]); 
}

__attribute__((reqd_work_group_size(3, 3, 3)))
__kernel void
dot_product333 (__global const float4 *a,  
	        __global const float4 *b, __global float *c) 
{ 
  int gid = get_global_id(0); 

  c[gid] = dot(a[gid], b[gid]); 
}
