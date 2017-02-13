#pragma OPENCL EXTENSION cl_khr_fp64 : enable

struct fakecomplex {
    double x;
      double y;
};

__kernel void sum(__global float *a, struct fakecomplex x)
{
      int gid = get_global_id(0);
          a[gid] = x.x;
}
