// Incompatible builtins must not leave F16C intrinsics in a Core 2 kernel.
kernel void
convert (global const float4 *src, global half *dst)
{
  size_t i = get_global_id (0);
  vstore_half4_rte (src[i], i, dst);
}
