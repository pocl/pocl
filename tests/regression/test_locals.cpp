/* Issues with __local pointers (lp:918801)

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "poclu.h"

#define WORK_ITEMS 2
#define BUFFER_SIZE (WORK_ITEMS)


static char
kernelSourceCode[] = 
"kernel void test_kernel (global float *a, local int *local_buf, private int scalar)\n"
"{\n"
"   int gid = get_local_id(0); \n"
"   local int automatic_local_scalar; \n"
"   local int automatic_local_buf[2];\n"
"\n"
"   __local int *p;\n"
"\n"
"   p = automatic_local_buf;\n"
"   p[gid] = gid + scalar;\n"
"   p = local_buf;\n"
"   p[gid] = a[gid];\n"
"   automatic_local_scalar = scalar;\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
"   a[gid] = automatic_local_buf[gid] + local_buf[gid] + automatic_local_scalar;\n"
"   \n"
"}\n";

int
main(void)
{
    float A[BUFFER_SIZE];

    try {
        std::vector<cl::Platform> platformList;

        // Pick platform
        cl::Platform::get(&platformList);

        // Pick first platform
        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, cprops);

        // Query the set of devices attched to the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create and program from source
        cl::Program::Sources sources({kernelSourceCode});
        cl::Program program(context, sources);

        cl_device_id dev_id = devices.at(0)();

        int scalar = poclu_bswap_cl_int (dev_id, 4);

        for (int i = 0; i < BUFFER_SIZE; ++i)
            A[i] = poclu_bswap_cl_float(dev_id, (cl_float)i);

        // Build program
        program.build(devices);

        cl::Buffer aBuffer = cl::Buffer(
            context, 
            CL_MEM_COPY_HOST_PTR,
            BUFFER_SIZE * sizeof(float), 
            (void *) &A[0]);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, aBuffer);
        kernel.setArg(1, (BUFFER_SIZE * sizeof(int)), NULL);
        kernel.setArg(2, scalar);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);
 
        // Do the work
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(WORK_ITEMS),
            cl::NDRange(WORK_ITEMS));
 
        // Map aBuffer to host pointer. This enforces a sync with 
        // the host backing space, remember we choose GPU device.
        float * res = (float *) queue.enqueueMapBuffer(
            aBuffer,
            CL_TRUE, // block 
            CL_MAP_READ,
            0,
            BUFFER_SIZE * sizeof(float));

        res[0] = poclu_bswap_cl_float (dev_id, res[0]);
        res[1] = poclu_bswap_cl_float (dev_id, res[1]);
        bool ok = res[0] == 8 && res[1] == 10;
        if (ok) {
            return EXIT_SUCCESS;
        } else {
            std::cout << "NOK " << res[0] << " " << res[1] << std::endl;
            std::cout << "res@" << std::hex << res << std::endl;
            return EXIT_FAILURE;
        }

        // Finally release our hold on accessing the memory
        queue.enqueueUnmapMemObject(
            aBuffer, (void *) res);
 
        // There is no need to perform a finish on the final unmap
        // or release any objects as this all happens implicitly with
        // the C++ Wrapper API.
    } 
    catch (cl::Error err) {
         std::cerr
             << "ERROR: "
             << err.what()
             << "("
             << err.err()
             << ")"
             << std::endl;

         return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
