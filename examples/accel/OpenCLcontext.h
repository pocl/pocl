
#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#undef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <mutex>
#include <memory>

class OpenCL_Context;

class OpenCL_Manager {
    bool isValid;
    std::unique_ptr<OpenCL_Context> Context;

public:

    OpenCL_Manager();
    ~OpenCL_Manager();

    bool initialize(unsigned width, unsigned height, unsigned bpp = 32);
    bool processCameraFrame(unsigned char* input, unsigned long *output);

};
