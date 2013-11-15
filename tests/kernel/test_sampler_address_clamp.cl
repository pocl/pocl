__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

kernel 
void test_sampler_address_clamp(__read_only image2d_t image)
{
  uint4 pixel = read_imageui(image, imageSampler, (int2)(0, 0));
  if(pixel.x != 1 || pixel.y !=1 || pixel.z != 1 || pixel.w != 1)
    {
      printf("in bounds read failed\n");
      printf("got: %d %d %d %d\n", pixel.x, pixel.y, pixel.z, pixel.w);
    }
  
  read_imageui(image, imageSampler, (int2)(3, 3));
  if(pixel.x != 1 || pixel.y !=1 || pixel.z != 1 || pixel.w != 1)
    printf("in bounds read failed\n");
  
  pixel = read_imageui(image, imageSampler, (int2)(-1, 0));
  if(pixel.x != 0|| pixel.y != 0 || pixel.z != 0 || pixel.w != 0)
    printf("out of bounds read failed\n");
  
  pixel = read_imageui(image, imageSampler, (int2)(0, -1));
  if(pixel.x != 0 || pixel.y != 0 || pixel.z != 0 || pixel.w != 0)
    printf("out of bounds read failed\n");
  
  pixel = read_imageui(image, imageSampler, (int2)(4, 0));
  if(pixel.x != 0 || pixel.y != 0 || pixel.z !=  0 || pixel.w != 0)
    printf("out of bounds read failed\n"); 
  
  pixel = read_imageui(image, imageSampler, (int2)(0, 4));
  if(pixel.x != 0 || pixel.y != 0 || pixel.z != 0 || pixel.w != 0)
    printf("out of bounds read failed\n");
  
}
