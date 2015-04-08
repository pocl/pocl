

kernel
void test_image_query_funcs(__read_only image2d_t image2, __read_only image3d_t image3)
{
  int h = get_image_height (image3);
  int w = get_image_width (image3);
  int d = get_image_depth (image3);
  int4 dim4 = get_image_dim (image3);

  if (h != 4)
    printf("get_image_height failed expected 4, got %d\n", h);

  if (w != 2)
    printf("get_image_width failed expected 2, got %d\n", w);

  if (d != 8)
    printf("get_image_width failed expected 8, got %d\n", d);

  if (dim4.x != w || dim4.y != h || dim4.z != d || dim4.w != 0)
    printf("get_image_dim failed expected (%d,%d,%d,0), got %v4d\n",
      w, h, d, dim4);

  h = get_image_height (image2);
  w = get_image_width (image2);
  int2 dim2 = get_image_dim (image2);

  if (h != 4)
    printf("get_image_height failed expected 4, got %d\n", h);

  if (w != 2)
    printf("get_image_width failed expected 2, got %d\n", w);

  if (dim2.x != w || dim2.y != h)
    printf("get_image_dim failed expected (%d,%d), got %v2d\n",
      w, h, dim2);

}
