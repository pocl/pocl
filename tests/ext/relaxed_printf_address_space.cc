/* relaxed_printf_address_space.cc -
   Test cases for cl_ext_relaxed_printf_address_space

   Copyright (c) 2022 Pekka Jääskeläinen / Parmance

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <CL/cl.h>

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

#define CHECK_ERR(X) do {					\
    if ((X) != CL_SUCCESS) {					\
      std::cerr << #X "ERROR at " << __LINE__ << std::endl;	\
      abort();							\
    } } while (0)

std::vector<std::string> getKernelNames(cl_program Program) {
  size_t KernelNamesSize;
  CHECK_ERR(clGetProgramInfo(Program, CL_PROGRAM_KERNEL_NAMES,
			     0, NULL,
			     &KernelNamesSize));


  char Names[KernelNamesSize + 1];
  CHECK_ERR(clGetProgramInfo(Program, CL_PROGRAM_KERNEL_NAMES,
			     KernelNamesSize, &Names[0],
			     &KernelNamesSize));

  // The specs doesn't ensure a null terminated string.
  Names[KernelNamesSize] = 0;

  std::vector<std::string> KernelNames;
  //Construct a stream from the string
  std::stringstream StreamData(Names);
  std::string Kern;
  while (std::getline(StreamData, Kern, ';')) {
    KernelNames.push_back(Kern);
  }
  return KernelNames;
}

int main(int Argc, char *Argv[]) {

  cl_uint Count;
  cl_platform_id Platform;

  clGetPlatformIDs (1, &Platform, &Count);
  if (Count == 0)
    abort();

  cl_context_properties Properties[] =
    {CL_CONTEXT_PLATFORM, (cl_context_properties)Platform, 0};

  cl_context Context =
    clCreateContextFromType (Properties, CL_DEVICE_TYPE_ALL,
			     NULL, NULL, NULL);

  cl_uint NumDevices;
  cl_int Err = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_ALL, 0, NULL,
			      &NumDevices);
  if (Err != CL_SUCCESS || NumDevices == 0)
    return EXIT_FAILURE;

  cl_device_id Device;
  Err = clGetDeviceIDs (Platform, CL_DEVICE_TYPE_ALL, 1, &Device, NULL);
  if (Err != CL_SUCCESS)
    return EXIT_FAILURE;

  std::ifstream File(SRCDIR "/relaxed_printf_address_space.cl",
		     std::ios::binary | std::ios::ate);
  std::streamsize SrcSize = File.tellg();
  File.seekg(0, std::ios::beg);

  std::vector<char> Src(SrcSize);
  if (!File.read(Src.data(), SrcSize))
    return EXIT_FAILURE;

  const char *SrcCStr = (const char*)Src.data();

  cl_program Program =
    clCreateProgramWithSource (Context, 1, &SrcCStr,
			       (const size_t*)&SrcSize, &Err);
  CHECK_ERR(Err);

  Err = clBuildProgram (Program, 0, NULL, "", NULL, NULL);
  CHECK_ERR(Err);

  cl_command_queue Queue =
    clCreateCommandQueueWithProperties(Context, Device, 0, &Err);
  CHECK_ERR(Err);

  cl_kernel Kernel = clCreateKernel(Program,
				    "relaxed_printf_address_space_tests",
				    NULL);
  int IO = 1;
  cl_mem IOBuffer =
    clCreateBuffer (Context,
		    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		    sizeof(int), &IO, &Err);
  CHECK_ERR(Err);
  Err = clSetKernelArg (Kernel, 0, sizeof (cl_mem), (void *)&IOBuffer);
  CHECK_ERR(Err);

  char GlobalFmtStr[] = "%s through a global format str\n\0";
  cl_mem GlobalFmtStrBuffer =
    clCreateBuffer (Context,
		    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		    strlen(GlobalFmtStr) + 1, &GlobalFmtStr, &Err);
  CHECK_ERR(Err);
  Err = clSetKernelArg (Kernel, 1, sizeof (cl_mem),
			(void *)&GlobalFmtStrBuffer);
  CHECK_ERR(Err);

  char GlobalStr[] = "global address space str arg\0";
  cl_mem GlobalStrBuffer =
    clCreateBuffer (Context,
		    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		    strlen(GlobalStr) + 1, &GlobalStr, &Err);
  CHECK_ERR(Err);
  Err = clSetKernelArg (Kernel, 2, sizeof (cl_mem),
			(void *)&GlobalStrBuffer);
  CHECK_ERR(Err);

  Err = clSetKernelArg (Kernel, 3, 1000, NULL);
  CHECK_ERR(Err);

  Err = clSetKernelArg (Kernel, 4, 1000, NULL);
  CHECK_ERR(Err);

  const size_t LocalSize[] = {1, 1, 1};
  const size_t GlobalSize[] = {1, 1, 1};

  Err = clEnqueueNDRangeKernel (Queue, Kernel, 1, NULL, LocalSize,
				GlobalSize, 0, NULL, NULL);
  CHECK_ERR(Err);

  Err = clEnqueueReadBuffer (Queue, IOBuffer, CL_TRUE, 0,
			       sizeof(int), &IO, 0, NULL, NULL);
  CHECK_ERR(Err);

  clFinish(Queue);

  std::cout << "IO == " << IO << std::endl << std::endl;

  return EXIT_SUCCESS;
}
