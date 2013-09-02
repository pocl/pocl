/* Macro definitions to create more easy main functions for running simple 
   regression test kernels.

   Copyright (c) 2013 Mikael Lepistö / Vincit
   
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

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iostream>
#include <cstring> // memcmp

// Returns true or false and in case of false outputs 
// expected and result to std::cerr
#define ASSERT_CHAR_PTR(expect, result, length) \
   do { \
   	if (memcmp(expect, result, length) == 0) return true; \
   	std::cerr << "Expected: "; \
   	for (unsigned i = 0; i < length; i++) {\
   		std::cerr << (int)expect[i] << ",";\
   	}\
   	std::cerr << std::endl;\
   	std::cerr << "Result: ";\
   	for (unsigned i = 0; i < length; i++) {\
   		std::cerr << (int)result[i] << ",";\
   	}\
   	std::cerr << std::endl;\
   	return false; \
   } while(0);

typedef struct {
	// pointer to data to pass buffer
	void *data;
	size_t dataSize;
	bool isBuffer;
	cl_mem_flags bufferFlags;
	cl::Buffer createdBuffer;
} KernelArg; 

// Macro to run simple kernels, without having to copy paste all the
// main program boilerplate.
//
// Requirements for the kernel to be tested is that it's first argument is 
// type of "__global char*" and all the values to be verified after running 
// should be written there.
//
// TODO: could be splitted to just some utility functions to keep 
//       debugging easier.
//
// @param source_code char* Null terminated string containing kernel source.
// @param kernel_name Name of kernel to run from source. e.g. "test_kernel"
// @param global_size Global work size as cl::NDRange object to pass 
//    enqueuNDRangeKenrel.
// @param local_size Local workgroup size (cl::NDRange) or cl:NullRange.
// @param check_output Function to call to verify output 
//    bool verifier(char* output, size_t count);
//    output table has output_size number of elements.
// @param output_size Number of elements to reserve to output table.
// @param args Rest of the kernel arguments as 
//    KernelArg table[] = { { data1, true, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR }, { } };
//

#define KERNEL_RUNNER(source_code, kernel_name, global_size, local_size, check_output, output_size, args ) \
int main() {                                                                          \
    char output[output_size] = { 0 };\
    cl_int err;\
    try {\
        std::vector<cl::Platform> platformList;\
        /* Pick platform */ \
        cl::Platform::get(&platformList); \
        /* Pick first platform */ \
        cl_context_properties cprops[] = {\
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};\
        cl::Context context(CL_DEVICE_TYPE_ALL, cprops);\
        /* Query the set of devices attched to the context */ \
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();\
        /* Create and program from source */ \
        cl::Program::Sources sources(1, std::make_pair(source_code, 0)); \
        cl::Program program(context, sources); \
        /* Build program */ \
        program.build(devices);\
        /* Create output buffer */ \
        cl::Buffer outputBuffer = cl::Buffer(\
            context, \
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, \
            output_size, \
            (void *) &output[0]);\
        /* Create kernel object */ \
        cl::Kernel kernel(program, kernel_name);\
        /* Set output arg */ \
        kernel.setArg(0, outputBuffer);\
        /* Create and set rest of kernel arguments */ \
        if (sizeof(args) > 0) { \
	        for (unsigned i = 0; \
   		     	i < sizeof(args)/sizeof(args[0]); \
        		i++) {\
        		KernelArg &arg = args[i];\
 				cl_int set_arg_err;\
        		if (arg.isBuffer) {\
    				cl_int err;\
       		 		arg.createdBuffer = \
       		 		  cl::Buffer(context, arg.bufferFlags, arg.dataSize, arg.data);\
        			set_arg_err = kernel.setArg(i+1, arg.createdBuffer);\
        		} else {\
       		 		set_arg_err = kernel.setArg(i+1, arg.dataSize, arg.data);\
        		}\
        		if (set_arg_err != CL_SUCCESS) {\
        			std::cerr << "Failed to set kernel arg: " << i+1 << std::endl;\
        			return EXIT_FAILURE;\
        		}\
	        }\
        }\
        /* Create command queue */ \
        cl::CommandQueue queue(context, devices[0], 0);\
        queue.enqueueNDRangeKernel(\
            kernel, \
            cl::NullRange, \
            global_size,\
            local_size);\
        /* Map outputBuffer to host pointer. This enforces a sync with */ \
        /* the host backing space, remember we choose GPU device. */\
        char* outputMap = (char *) queue.enqueueMapBuffer(\
            outputBuffer,\
            CL_TRUE, /*block*/ \
            CL_MAP_READ,\
            0, output_size);\
        bool ok = check_output (outputMap, output_size);\
        /* Finally release our hold on accessing the memory */\
        err = queue.enqueueUnmapMemObject(\
            outputBuffer,\
            (void *) outputMap);\
        /* There is no need to perform a finish on the final unmap */\
        /* or release any objects as this all happens implicitly with */\
        /* the C++ Wrapper API. */\
        if (ok) \
          return EXIT_SUCCESS;\
        else\
          return EXIT_FAILURE;\
    } \
    catch (cl::Error err) {\
         std::cerr\
             << "ERROR: "\
             << err.what()\
             << "("\
             << err.err()\
             << ")"\
             << std::endl;\
         return EXIT_FAILURE;\
    }\
    return EXIT_SUCCESS;\
}

