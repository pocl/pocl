__attribute__((__overloadable__))
float dot(float p0, float p1)
{
  return p0*p1;
}

__attribute__((__overloadable__))
float dot(float2 p0, float2 p1)
{
  return p0.x*p1.x + p0.y*p1.y;
}

__attribute__((__overloadable__))
float dot(float3 p0, float3 p1)
{
  return p0.x*p1.x + p0.y*p1.y + p0.z*p1.z;
}

__attribute__((__overloadable__))
float dot(float4 p0, float4 p1)
{
  return p0.x*p1.x + p0.y*p1.y + p0.z*p1.z + p0.w*p1.w;
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double dot(double p0, double p1)
{
  return p0*p1;
}

__attribute__((__overloadable__))
double dot(double2 p0, double2 p1)
{
  return p0.x*p1.x + p0.y*p1.y;
}

__attribute__((__overloadable__))
double dot(double3 p0, double3 p1)
{
  return p0.x*p1.x + p0.y*p1.y + p0.z*p1.z;
}

__attribute__((__overloadable__))
double dot(double4 p0, double4 p1)
{
  return p0.x*p1.x + p0.y*p1.y + p0.z*p1.z + p0.w*p1.w;
}
#endif
