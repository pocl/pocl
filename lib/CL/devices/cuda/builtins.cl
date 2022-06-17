__kernel void
pocl_abs_f32 (global const float *__restrict input,
              global float *__restrict output)
{
  size_t i = get_global_id (0);
  output[i] = fabs (input[i]);
}

__kernel void
org_khronos_openvx_scale_image_nn_u8 (
    global const unsigned char *__restrict input,
    global unsigned char *__restrict output, float width_scale,
    float height_scale, int input_width, int input_height)
{
  int x = get_global_id (0);
  int y = get_global_id (1);
  int z = get_global_id (2);
  int num_planes = get_global_size (2);
  // printf("x=%d,y=%d",x,y);

  float x_src = ((float)x + 0.5f) * width_scale - 0.5f;
  float y_src = ((float)y + 0.5f) * height_scale - 0.5f;
  // printf("x_s=%f,y_s=%f",x_src,y_src);
  float x_min = floor (x_src);
  float y_min = floor (y_src);
  // printf("x_m=%f,y_m=%f",x_min,y_min);

  int x1 = (int)x_min;
  int y1 = (int)y_min;
  // printf("x1=%d,y1=%d",x1,y1);

  if (x_src - x_min >= 0.5f)
    x1++;
  if (y_src - y_min >= 0.5f)
    y1++;

  unsigned char result = input[num_planes * (x1 * input_width + y1) + z];
  // printf("x1=%d,y1=%d",x1,y1);

  // printf("result=%d",result);

  output[num_planes * (x * get_global_size (0) + y) + z] = result;

  // for(int i =0;i<9;i++) {
  //  printf("input value %d\n",input[i]);
  //}
}

__kernel void
org_khronos_openvx_scale_image_bl_u8 (
    global const unsigned char *__restrict input,
    global unsigned char *__restrict output, float width_scale,
    float height_scale, int input_width, int input_height)
{
  size_t planes = get_global_size (2);
  size_t x = get_global_id (0);
  size_t y = get_global_id (1);
  size_t plane = get_global_id (2);

  // printf("x=%d,y=%d",x,y);
  int input_size = input_width * input_height;
  int output_size = get_global_size (0) * get_global_size (1);

  float x_src = ((float)x + 0.5f) * width_scale - 0.5f;
  float y_src = ((float)y + 0.5f) * height_scale - 0.5f;
  // printf("x_s=%f,y_s=%f",x_src,y_src);
  int x_min = floor (x_src);
  int y_min = floor (y_src);
  // printf("x_m=%f,y_m=%f",x_min,y_min);

  float s = x_src - x_min;
  float t = y_src - y_min;

  float result
      = (1 - s) * (1 - t)
            * input[planes * (x_min * input_width + y_min) + plane]
        + s * (1 - t)
              * input[planes * ((x_min + 1) * input_width + y_min) + plane]
        + (1 - s) * t
              * input[planes * (x_min * input_width + y_min + 1) + plane]
        + s * t
              * input[planes * ((x_min + 1) * input_width + y_min + 1)
                      + plane];
  // printf("x1=%d,y1=%d",x1,y1);

  // printf("result=%d",result);

  output[planes * (x * get_global_size (0) + y) + plane]
      = (unsigned char)result;

  // printf("%hhu", (unsigned char)result);
  // for(int i =0;i<9;i++) {
  //  printf("input value %d\n",input[i]);
  //}
}

__kernel void
pocl_add_i8 (global const signed char *__restrict a,
             global const signed char *__restrict b,
             global signed char *__restrict output)
{
  size_t x = get_global_id (0);

  output[x] = a[x] + b[x];
}
