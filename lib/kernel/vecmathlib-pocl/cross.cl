__attribute__((__overloadable__))
float4 cross(float4 p0, float4 p1)
{
  float4 r;
  r.x = p0.y*p1.z - p0.z*p1.y;
  r.y = p0.z*p1.x - p0.x*p1.z;
  r.z = p0.x*p1.y - p0.y*p1.x;
  r.w = 0.0f;
  return r;
}

__attribute__((__overloadable__))
float3 cross(float3 p0, float3 p1)
{
  float3 r;
  r.x = p0.y*p1.z - p0.z*p1.y;
  r.y = p0.z*p1.x - p0.x*p1.z;
  r.z = p0.x*p1.y - p0.y*p1.x;
  return r;
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double4 cross(double4 p0, double4 p1)
{
  double4 r;
  r.x = p0.y*p1.z - p0.z*p1.y;
  r.y = p0.z*p1.x - p0.x*p1.z;
  r.z = p0.x*p1.y - p0.y*p1.x;
  r.w = 0.0f;
  return r;
}

__attribute__((__overloadable__))
double3 cross(double3 p0, double3 p1)
{
  double3 r;
  r.x = p0.y*p1.z - p0.z*p1.y;
  r.y = p0.z*p1.x - p0.x*p1.z;
  r.z = p0.x*p1.y - p0.y*p1.x;
  return r;
}
#endif
