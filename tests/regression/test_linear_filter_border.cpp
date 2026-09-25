/* Regression test: with CLK_ADDRESS_CLAMP, the border color of an image
   whose channel order has no alpha (CL_R, CL_RG) is (0,0,0,1), so a linear
   read that touches the border must return alpha 1. The linear filters used
   to skip out-of-range texels, which returned alpha 0 for a read entirely
   outside the image and a fraction of 1 for one straddling its edge. The
   OpenCL CTS (images/kernel_read_write) catches this, but only on builds
   without ENABLE_CONFORMANCE: conformance builds don't offer CL_R or CL_RG,
   and this test skips there for the same reason.

   Copyright (c) 2026 PoCL developers. MIT license, see COPYING. */

#include "pocl_opencl.h"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *SOURCE = R"RAW(
__constant sampler_t smp
    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__kernel void k(read_only image2d_t img, __global const float2 *coord,
                __global float4 *out) {
  size_t i = get_global_id(0);
  out[i] = read_imagef(img, smp, coord[i]);
}
)RAW";

static bool supported(const cl::Context &ctx, cl_channel_order order) {
  std::vector<cl::ImageFormat> formats;
  ctx.getSupportedImageFormats(CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D,
                               &formats);
  for (const cl::ImageFormat &f : formats)
    if (f.image_channel_order == order
        && f.image_channel_data_type == CL_FLOAT)
      return true;
  return false;
}

int main() {
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  const cl_channel_order orders[2] = {CL_R, CL_RG};
  const char *names[2] = {"CL_R", "CL_RG"};
  if (!supported(context, CL_R) && !supported(context, CL_RG)) {
    printf("SKIP: the device offers neither CL_R nor CL_RG with CL_FLOAT\n");
    return 77;
  }

  cl::Program program(context, SOURCE);
  program.build();
  cl::Kernel kernel(program, "k");

  /* A 4x4 image, every channel 0.5. Unnormalized coordinates: texel i
     spans [i, i+1), so x = 0.25 blends texel -1 (the border) and texel 0
     with weights 0.25 and 0.75. */
  const int W = 4, H = 4;
  const size_t N = 4;
  const cl_float2 coord[N] = {{{-9.0f, 2.0f}},   /* entirely outside */
                              {{20.0f, 20.0f}},  /* entirely outside */
                              {{0.25f, 2.0f}},   /* straddles the edge */
                              {{2.0f, 2.0f}}};   /* inside */
  /* expected red: border 0, texels 0.5 */
  const float red[N] = {0.0f, 0.0f, 0.75f * 0.5f, 0.5f};

  int failures = 0;
  for (int o = 0; o < 2; ++o) {
    if (!supported(context, orders[o]))
      continue;
    const int nch = orders[o] == CL_R ? 1 : 2;
    std::vector<float> texels(W * H * nch, 0.5f);
    cl::Image2D img(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    cl::ImageFormat(orders[o], CL_FLOAT), W, H, 0,
                    texels.data());
    cl::Buffer cbuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(coord), (void *)coord);
    cl::Buffer obuf(context, CL_MEM_WRITE_ONLY, N * sizeof(cl_float4));
    kernel.setArg(0, img);
    kernel.setArg(1, cbuf);
    kernel.setArg(2, obuf);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
    cl_float4 out[N];
    queue.enqueueReadBuffer(obuf, CL_TRUE, 0, sizeof(out), out);

    for (size_t i = 0; i < N; ++i) {
      const float green = nch == 2 ? red[i] : 0.0f;
      bool ok = std::fabs(out[i].s[3] - 1.0f) < 1e-6f
                && std::fabs(out[i].s[0] - red[i]) < 1e-3f
                && std::fabs(out[i].s[1] - green) < 1e-3f
                && out[i].s[2] == 0.0f;
      if (!ok) {
        printf("%s at (%g, %g): got (%g, %g, %g, %g), expected (%g, %g, 0, 1)\n",
               names[o], coord[i].s[0], coord[i].s[1], out[i].s[0],
               out[i].s[1], out[i].s[2], out[i].s[3], red[i], green);
        ++failures;
      }
    }
  }

  if (failures) {
    printf("FAIL: %d reads\n", failures);
    return EXIT_FAILURE;
  }
  printf("OK\n");
  return EXIT_SUCCESS;
}
