/*
  Github Issue #1435
*/

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <cassert>
#include <iostream>

using namespace std;

const char *SOURCE = R"RAW(

__kernel void medfilt2d(__global float *image,  // input image
                        __global float *result, // output array
                        __local  float4 *l_data,// local storage 4x the number of threads
                                 int khs1,      // Kernel half-size along dim1 (nb lines)
                                 int khs2,      // Kernel half-size along dim2 (nb columns)
                                 int height,    // Image size along dim1 (nb lines)
                                 int width)     // Image size along dim2 (nb columns)
{
    int threadid = get_local_id(0);
    int x = get_global_id(1);

    if (x < width)
    {
        union
        {
            float  ary[8];
            float8 vec;
        } output, input;
        input.vec = (float8)(MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT);
        int kfs1 = 2 * khs1 + 1; 
        int kfs2 = 2 * khs2 + 1;
        int nbands = (kfs1 + 7) / 8; 
        for (int y=0; y<height; y++)
        {
            //Select only the active threads, some may remain inactive
            int nb_threads =  (nbands * kfs2);
            int band_nr = threadid / kfs2;
            int band_id = threadid % kfs2;
            int pos_x = clamp((int)(x + band_id - khs2), (int) 0, (int) width-1);
            int max_vec = clamp(kfs1 - 8 * band_nr, 0, 8);
            if (y == 0)
            {
                for (int i=0; i<max_vec; i++)
                {
                    if (threadid<nb_threads)
                    {
                        int pos_y = clamp((int)(y + 8 * band_nr + i - khs1), (int) 0, (int) height-1);
                        input.ary[i] = image[pos_x + width * pos_y];
                    }
                }
            }
            else
            {
                //store storage.s0 to some shared memory to retrieve it from another thread.
                l_data[threadid].s0 = input.vec.s0;

                //Offset to the bottom
                input.vec = (float8)(input.vec.s1,
                        input.vec.s2,
                        input.vec.s3,
                        input.vec.s4,
                        input.vec.s5,
                        input.vec.s6,
                        input.vec.s7,
                        MAXFLOAT);

                barrier(CLK_LOCAL_MEM_FENCE);

                int read_from = threadid + kfs2;
                if (read_from < nb_threads)
                    input.vec.s7 = l_data[read_from].s0;
                else if (threadid < nb_threads) //we are on the last band
                {
                    int pos_y = clamp((int)(y + 8 * band_nr + max_vec - 1 - khs1), (int) 0, (int) height-1);
                    input.ary[max_vec - 1] = image[pos_x + width * pos_y];
                }

            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

)RAW";

#if 0

// the shorter code that should trigger the same issue

const char *SOURCE = R"RAW(

__kernel void testkernel(__local float2 *b) {
  struct {
    int c[1];
    float2 d;
  } e;
  for (int f = 0; f < 2; f++) {
    if (f)
      for (int g; g < (int)b[0].x; g++)
        e.c[g] = 0;
    else if (b)
      e.d.s0 = b[0].s0;
    barrier(0);
  }
}
)RAW";

#endif

int main(int argc, char *argv[]) {
  cl::Device device = cl::Device::getDefault();
  cl::Program program(SOURCE);
  program.build("-cl-std=CL1.2");

  // This triggers compilation of dynamic WG binaries.
  cl::Program::Binaries binaries{};
  int err = program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);
  if (err == CL_SUCCESS) {
    printf("OK\n");
    return EXIT_SUCCESS;
  } else {
    printf("FAIL\n");
    return EXIT_FAILURE;
  }
}
