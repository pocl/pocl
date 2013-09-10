kernel void test_kernel() {
  printf("global: %d-%d-%d\n",
           (int)get_global_id(0),
           (int)get_global_id(1),
           (int)get_global_id(2));
  printf("local: %d-%d-%d\n",
           (int)get_local_id(0),
           (int)get_local_id(1),
           (int)get_local_id(2));
}
