#define PADDING         (32) 
#define GROUP_DIMX      (32) 
#define LOG_GROUP_DIMX  (5) 
#define GROUP_DIMY      (2) 
#define WIDTH           (256) 
#define HEIGHT          (4096) 

int printf(const char *restrict format, ...);
 
__kernel void
matrix_transpose(__global float *output, 
		 __global float *input)
{
  __local float tile[(32 + 1) * 32];

  /* __local float *volatile tmp = &tile[0]; */

  /* printf("gid=(%d,%d), output=%p, input=%p, tile=%p\n", */
  /*        get_global_id(1), get_global_id(0), */
  /*        output, input, tmp); */

  int block_x = get_group_id(0); 
  int block_y = get_group_id(1); 
  
  int local_x = get_local_id(0) & (GROUP_DIMX - 1); 
  int local_y = get_local_id(0) >> LOG_GROUP_DIMX; 
  
  int local_input  = mad24(local_y, GROUP_DIMX + 1, local_x); 
  int local_output = mad24(local_x, GROUP_DIMX + 1, local_y); 
  
  int in_x = mad24(block_x, GROUP_DIMX, local_x); 
  int in_y = mad24(block_y, GROUP_DIMX, local_y);  
  
  int input_index = mad24(in_y, WIDTH, in_x);  
  
  int out_x = mad24(block_y, GROUP_DIMX, local_x); 
  int out_y = mad24(block_x, GROUP_DIMX, local_y);  
  
  int output_index = mad24(out_y, HEIGHT + PADDING, out_x);  
  
  int global_input_stride  = WIDTH * GROUP_DIMY; 
  int global_output_stride = (HEIGHT + PADDING) * GROUP_DIMY; 
  
  int local_input_stride  = GROUP_DIMY * (GROUP_DIMX + 1); 
  int local_output_stride = GROUP_DIMY; 

  tile[local_input] = input[input_index]; 
  local_input += local_input_stride;  
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index]; 
  local_input += local_input_stride; 
  input_index += global_input_stride; 
  
  tile[local_input] = input[input_index];  
  
  barrier(CLK_LOCAL_MEM_FENCE); 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
  
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
 
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
 
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
 
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
 
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
 
  output[output_index] = tile[local_output]; 
  local_output += local_output_stride; 
  output_index += global_output_stride; 
 
  output[output_index] = tile[local_output];  
}
