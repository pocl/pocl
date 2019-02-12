__kernel void
test_kernel (global int *output)
{
  size_t flat_id =
    get_global_id (2) * get_global_size (1) +
    get_global_id (1) * get_global_size (0) +
    get_global_id (0);

  switch (flat_id) {
  case 1:
    output[flat_id] = 101;
    return;
  case 3:
    output[flat_id] = 303;
    break;
  default:
    output[flat_id] = 99;
  }
}
