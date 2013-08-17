__attribute__((__overloadable__))
float fast_distance(float p0, float p1)
{
  return fast_length(p0-p1);
}

__attribute__((__overloadable__))
float fast_distance(float2 p0, float2 p1)
{
  return fast_length(p0-p1);
}

__attribute__((__overloadable__))
float fast_distance(float3 p0, float3 p1)
{
  return fast_length(p0-p1);
}

__attribute__((__overloadable__))
float fast_distance(float4 p0, float4 p1)
{
  return fast_length(p0-p1);
}
