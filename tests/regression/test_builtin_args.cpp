#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iostream>

using namespace std;

int main(int, char **)
{
  try {
    int N = 9;

    cl::Platform platform;
    std::vector<cl::Device> devices;

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(!all_platforms.size()) {
        std::cerr << "No OpenCL platforms available!\n";
        return 1;
    }
    platform = all_platforms[0];

    cl::Device FpgaDev;
    cl::CommandQueue FpgaQueue;
    cl::Program FpgaProgram;
    cl::Kernel FpgaKernel;

    // Find all devices.
    platform.getDevices(CL_DEVICE_TYPE_CUSTOM, &devices);
    if(devices.size() == 0) {
        std::cerr << "No OpenCL 'custom' devices available!\n";
        return 2;
    }
    FpgaDev = devices[0];
    std::vector<cl::Device> FpgaDevs = {FpgaDev};
    cl::Context ClContext{FpgaDevs};

    std::string FpgaBuiltinKernels = FpgaDev.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
    const std::string BuiltinKernelName{"pocl.countred"};
    FpgaProgram = cl::Program{ClContext, FpgaDevs, BuiltinKernelName};
    FpgaProgram.build(FpgaDevs);
    FpgaKernel = cl::Kernel(FpgaProgram, BuiltinKernelName.c_str());

    // *****************************************************************

    std::string kernel_name = FpgaKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
    std::string a = FpgaKernel.getInfo<CL_KERNEL_ATTRIBUTES>();
    unsigned num_args = FpgaKernel.getInfo<CL_KERNEL_NUM_ARGS>();

    for (cl_uint arg_index = 0; arg_index < num_args; ++arg_index) {

        cl_kernel_arg_access_qualifier acc_q = FpgaKernel.getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index);
        cl_kernel_arg_address_qualifier addr_q = FpgaKernel.getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index);
        cl_kernel_arg_type_qualifier type_q = FpgaKernel.getArgInfo<CL_KERNEL_ARG_TYPE_QUALIFIER>(arg_index);

        std::string arg_typename = FpgaKernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index);
        std::string arg_name = FpgaKernel.getArgInfo<CL_KERNEL_ARG_NAME>(arg_index);

        std::cerr << "KERNEL " << kernel_name << " | ARG " << arg_index << " | NAME " << arg_name << " | TYPE "
                  << arg_typename << " | ACC Q " << acc_q << " | ADDR Q "
                  << addr_q << " | TYPE Q " << type_q << "\n";
    }
  }

  catch (cl::Error& err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << " what: " << err.what() << std::endl;
    return 11;
  }

  return 0;
}
