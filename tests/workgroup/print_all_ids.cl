kernel void test_kernel() {
  printf("global: %zd-%zd-%zd\n",
           get_global_id(0),
           get_global_id(1),
           get_global_id(2));
  printf("local: %zd-%zd-%zd\n",
           get_local_id(0),
           get_local_id(1),
           get_local_id(2));
}

