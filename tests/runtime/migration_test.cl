kernel void
migration_test (global const float *in, global float *out, uint items,
                uint device_index)
{
  size_t i = get_global_id (0);
  const size_t offset = device_index * items;
  out[offset + i] += in[offset + i] * (device_index + 1);
  if (i == 0)
    printf ("\nDEV: %u OUT[0]: %f IN[0]: %f \n\n", device_index,
            out[offset + i], in[offset + i]);
}
