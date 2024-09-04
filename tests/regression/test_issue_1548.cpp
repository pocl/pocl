#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#define ASSERT_SUCCESS(_exp)                                                   \
  do {                                                                         \
    if ((_exp) != CL_SUCCESS) {                                                \
      std::cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

template <typename FnTy>
FnTy LoadOclExtFn(cl::Platform &P, const std::string &FnName) {
  auto PId = P.Wrapper<cl_platform_id>::get();
  auto TheFn = reinterpret_cast<FnTy>(
      clGetExtensionFunctionAddressForPlatform(PId, FnName.c_str()));
  if (!TheFn) {
    std::cerr << "Could not find platform extension: " << FnName
              << ";\nTest SKIPPED\n";
    exit(77);
  }
  return TheFn;
}

int main() try {
  unsigned PlatformIdx = 0;

  std::vector<cl::Platform> Platforms;
  cl::Platform::get(&Platforms);
  cl::Platform Platform = Platforms.at(PlatformIdx);
  std::cout << "Selected platform: " << Platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl;

  std::vector<cl::Device> Devices;
  Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
  unsigned DevIdx = 0;
  for (auto &Dev : Devices)
    std::cout << "Devices[" << DevIdx++
              << "]: " << Dev.getInfo<CL_DEVICE_NAME>() << std::endl;
  if (Devices.size() < 2) {
    std::cerr << "This test requires at least two devices with USM support\n";
    exit(77);
  }

  auto AllocDevMem =
      LoadOclExtFn<clDeviceMemAllocINTEL_fn>(Platform, "clDeviceMemAllocINTEL");
  auto FreeMem = LoadOclExtFn<clMemFreeINTEL_fn>(Platform, "clMemFreeINTEL");

  auto Ctx = cl::Context(Devices);

  cl_int Err;

  void *Dev0Mem = AllocDevMem(Ctx.get(), Devices[0].get(), nullptr, 8, 8, &Err);
  ASSERT_SUCCESS(Err);
  Err = FreeMem(Ctx.get(), Dev0Mem);
  ASSERT_SUCCESS(Err);

  void *Dev1Mem = AllocDevMem(Ctx.get(), Devices[1].get(), nullptr, 8, 8, &Err);
  ASSERT_SUCCESS(Err);
  Err = FreeMem(Ctx.get(), Dev1Mem);
  ASSERT_SUCCESS(Err);

  return 0;
} catch (cl::Error &Ex) {
  std::cerr << "Caught OpenCL exception: " << Ex.what() << "\n";
  std::cerr << "Error code: " << Ex.err() << "\n";
  return 2;
} catch (std::exception &Ex) {
  std::cerr << "Caught STL exception: " << Ex.what() << "\n";
  return 2;
}
