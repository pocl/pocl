/* Regression test: clEnqueueCopyImage and clEnqueueCopyBufferRect given a NULL
   memory object must return CL_INVALID_MEM_OBJECT. pocl_validate_rect_copy
   compared src->context and dst->context with the queue's before checking that
   src and dst are valid objects, so a NULL one crashed. OpenCL-CTS
   images/clCopyImage (test_copy_1D_buffer) reaches this on a device whose
   CL_DEVICE_IMAGE_MAX_BUFFER_SIZE exceeds 2^31.

   Copyright (c) 2026 PoCL developers. MIT license, see COPYING. */

#include "pocl_opencl.h"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <cstdio>
#include <cstdlib>

static int check(const char *what, cl_int got) {
  bool ok = got == CL_INVALID_MEM_OBJECT;
  std::printf("%s: %d (%s)\n", what, got,
              ok ? "CL_INVALID_MEM_OBJECT" : "want CL_INVALID_MEM_OBJECT");
  return ok ? 0 : 1;
}

int main() {
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);
  cl_command_queue q = queue();
  const size_t origin[3] = {0, 0, 0}, region[3] = {1, 1, 1};
  int bad = 0;

  cl::Buffer buf(context, CL_MEM_READ_WRITE, 64);
  bad += check("clEnqueueCopyBufferRect(NULL src)",
               clEnqueueCopyBufferRect(q, nullptr, buf(), origin, origin,
                                       region, 0, 0, 0, 0, 0, nullptr,
                                       nullptr));
  bad += check("clEnqueueCopyBufferRect(NULL dst)",
               clEnqueueCopyBufferRect(q, buf(), nullptr, origin, origin,
                                       region, 0, 0, 0, 0, 0, nullptr,
                                       nullptr));

  if (device.getInfo<CL_DEVICE_IMAGE_SUPPORT>()) {
    cl::Image2D img(context, CL_MEM_READ_WRITE,
                    cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), 4, 4);
    bad += check("clEnqueueCopyImage(NULL src)",
                 clEnqueueCopyImage(q, nullptr, img(), origin, origin, region,
                                    0, nullptr, nullptr));
    bad += check("clEnqueueCopyImage(NULL dst)",
                 clEnqueueCopyImage(q, img(), nullptr, origin, origin, region,
                                    0, nullptr, nullptr));
  }

  queue.finish();
  if (bad) {
    std::printf("FAIL: %d call(s) did not return CL_INVALID_MEM_OBJECT\n", bad);
    return EXIT_FAILURE;
  }
  std::printf("OK\n");
  return EXIT_SUCCESS;
}
