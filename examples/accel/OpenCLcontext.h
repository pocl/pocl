
#pragma once

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
