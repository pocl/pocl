/* Tests failure of initializing table inside struct.

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

#include "kernel_runner.hpp"


static char
kernelSourceCode[] =
"typedef struct {\n"
"    float table[3];\n"
"} TempStruct;\n"
"\n"
"typedef struct {\n"
"    float single;\n"
"} TempStruct2;\n"
"\n"
"__constant TempStruct constant_struct = { {0, 1, 2} };\n"
"__kernel void struct_test(__global char* output, __global float *test) {\n"
"    size_t gid = get_global_id(0);\n"
"    TempStruct private_struct = { {4, 5, 6} };\n"
"    TempStruct2 private_struct2 = { 7 };\n"
"    output[gid*3] = constant_struct.table[gid];\n"
"    output[gid*3+1] = private_struct.table[gid];\n"
"    output[gid*3+2] = private_struct2.single;\n"
"}\n";

bool check_results(char* output, size_t count) {
	char valid[] = {0,4,7,1,5,7,2,6,7};
	// returns true or false and output debug info
	ASSERT_CHAR_PTR(valid, output, count);
}

// extra argument just to test kernel runner implementation
float testArg[] = {1,2,3}; 
KernelArg kernelArgs[] = {
	{ 
	  testArg, // data
	  sizeof(testArg), // data size in bytes 
	  true,  // is buffer
	  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // buffer flags
	  NULL // this will be initialized with created buffer
	}
};

KERNEL_RUNNER(kernelSourceCode, 
	"struct_test", cl::NDRange(3), cl::NullRange, 
	check_results, 9, kernelArgs)
