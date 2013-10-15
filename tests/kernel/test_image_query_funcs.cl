

kernel 
void test_image_query_funcs(__read_only image3d_t image)
{
  int h = get_image_height (image);
  int w = get_image_width (image);
  int d = get_image_depth (image);
  
  if (h != 4)
    printf("get_image_height failed expected 4, got %d\n", h);

  if (w != 2)
    printf("get_image_width failed expected 2, got %d\n", w);

  if (d != 8)
    printf("get_image_width failed expected 8, got %d\n", d);  

}
