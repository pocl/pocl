#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void 
dot_product (__global const float4 *a,  
	     __global const float4 *b, __global float *c) 
{ 
  int gid = get_global_id(0); 

  /* This parallel region does not vectorize with the
     loop vectorizer because it accesses vector datatypes.
     Perhaps with SLP/BB vectorizer.*/

  float ax = a[gid].x;
  float ay = a[gid].y; 
  float az = a[gid].z;
  float aw = a[gid].w;

  float bx = b[gid].x, 
      by = b[gid].y, 
      bz = b[gid].z, 
      bw = b[gid].w;
  
  barrier(CLK_LOCAL_MEM_FENCE);

  /* This parallel region should vectorize. */
  c[gid] = ax * bx;
  c[gid] += ay * by;
  c[gid] += az * bz;
  c[gid] += aw * bw;
}
