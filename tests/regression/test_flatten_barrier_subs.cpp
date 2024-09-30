/* Tests a kernel create with binary.

   Copyright (c) 2018 Julius Ikkala / TUT

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

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <iostream>
#include <numeric>

#ifdef _WIN32
#  include "vccompat.hpp"
#endif

std::string read_text_file(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");

    if(!f)
    {
        throw std::runtime_error("Unable to open " + path);
    }

    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* data = new char[sz];
    if(fread(data, 1, sz, f) != sz)
    {
        delete [] data;
        throw std::runtime_error("Unable to read " + path);
    }
    fclose(f);
    std::string ret(data, sz);

    delete [] data;
    return ret;
}

cl::Platform get_platform(unsigned force_platform = 0)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty())
        throw std::runtime_error("No platforms found!");

    for(unsigned i = 0; i < platforms.size(); ++i)
    {
        cl::Platform& p = platforms[i];
        std::string name = p.getInfo<CL_PLATFORM_NAME>();
        std::cout << i << ": " << name << std::endl;
    }

    if(force_platform >= platforms.size()) force_platform = 0;

    return platforms[force_platform];
}

cl::Device get_device(cl::Platform pl, unsigned force_device = 0)
{
    std::vector<cl::Device> devices;
    pl.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.empty())
        throw std::runtime_error("No devices found!");

    return devices[0];
}

void exclusive_scan_cpu(const std::vector<int>& input, std::vector<int>& output)
{
    output.resize(input.size());
    int a = 0;
    for(unsigned i = 0; i < input.size(); ++i)
    {
        output[i] = a;
        a += input[i];
    }
}

void exclusive_scan_cl(const std::vector<int>& input, std::vector<int>& output)
{
    // Fails on POCL, works with AMDGPU-PRO
    cl::Platform p = get_platform(0);
    cl::Device d = get_device(p);
    try {
        cl::Context ctx(d);
        std::string src =
            read_text_file(SRCDIR "/test_flatten_barrier_subs.cl");
        cl::Program::Sources sources;
        sources.push_back({src.c_str(), src.length()});
        cl::Program program(ctx, sources);
        program.build({d});

        cl::Buffer input_buf(ctx, CL_MEM_READ_WRITE,
                             sizeof(int) * input.size());
        cl::Buffer output_buf(ctx, CL_MEM_READ_WRITE,
                              sizeof(int) * input.size());
        cl::CommandQueue q(ctx, d);
        q.enqueueWriteBuffer(input_buf, CL_TRUE, 0, sizeof(int) * input.size(),
                             input.data());

        int numElems = input.size();
#define WG_SIZE 64
    int GROUP_BLOCK_SIZE_SCAN = (WG_SIZE << 3);
    int GROUP_BLOCK_SIZE_DISTRIBUTE = (WG_SIZE << 2);

    int NUM_GROUPS_BOTTOM_LEVEL_SCAN = (numElems + GROUP_BLOCK_SIZE_SCAN - 1) / GROUP_BLOCK_SIZE_SCAN;
    int NUM_GROUPS_MID_LEVEL_SCAN = (NUM_GROUPS_BOTTOM_LEVEL_SCAN + GROUP_BLOCK_SIZE_SCAN - 1) / GROUP_BLOCK_SIZE_SCAN;
    int NUM_GROUPS_TOP_LEVEL_SCAN = (NUM_GROUPS_MID_LEVEL_SCAN + GROUP_BLOCK_SIZE_SCAN - 1) / GROUP_BLOCK_SIZE_SCAN;

    int NUM_GROUPS_BOTTOM_LEVEL_DISTRIBUTE = (numElems + GROUP_BLOCK_SIZE_DISTRIBUTE - 1) / GROUP_BLOCK_SIZE_DISTRIBUTE;
    int NUM_GROUPS_MID_LEVEL_DISTRIBUTE = (NUM_GROUPS_BOTTOM_LEVEL_DISTRIBUTE + GROUP_BLOCK_SIZE_DISTRIBUTE - 1) / GROUP_BLOCK_SIZE_DISTRIBUTE;

    cl::Buffer devicePartSumsBottomLevel(
        ctx, CL_MEM_READ_WRITE, sizeof(int)*NUM_GROUPS_BOTTOM_LEVEL_SCAN
    );
    cl::Buffer devicePartSumsMidLevel(
        ctx, CL_MEM_READ_WRITE, sizeof(int)*NUM_GROUPS_MID_LEVEL_SCAN
    );
    cl::Kernel bottomLevelScan(program, "scan_exclusive_part_int4");
    cl::Kernel topLevelScan(program, "scan_exclusive_int4");
    cl::Kernel distributeSums(program, "distribute_part_sum_int4");

    bottomLevelScan.setArg(0, input_buf);
    bottomLevelScan.setArg(1, output_buf);
    bottomLevelScan.setArg(2, numElems);
    bottomLevelScan.setArg(3, devicePartSumsBottomLevel);
    bottomLevelScan.setArg(4, WG_SIZE * sizeof(cl_int), nullptr);
    q.enqueueNDRangeKernel(
        bottomLevelScan,
        cl::NullRange,
        cl::NDRange(NUM_GROUPS_BOTTOM_LEVEL_SCAN * WG_SIZE),
        cl::NDRange(WG_SIZE)
    );

    bottomLevelScan.setArg(0, devicePartSumsBottomLevel);
    bottomLevelScan.setArg(1, devicePartSumsBottomLevel);
    bottomLevelScan.setArg(2, (cl_uint)NUM_GROUPS_BOTTOM_LEVEL_SCAN);
    bottomLevelScan.setArg(3, devicePartSumsMidLevel);
    bottomLevelScan.setArg(4, WG_SIZE * sizeof(cl_int), nullptr);
    q.enqueueNDRangeKernel(
        bottomLevelScan,
        cl::NullRange,
        cl::NDRange(NUM_GROUPS_MID_LEVEL_SCAN * WG_SIZE),
        cl::NDRange(WG_SIZE)
    );

    topLevelScan.setArg(0, devicePartSumsMidLevel);
    topLevelScan.setArg(1, devicePartSumsMidLevel);
    topLevelScan.setArg(2, (cl_uint)NUM_GROUPS_MID_LEVEL_SCAN);
    topLevelScan.setArg(3, WG_SIZE * sizeof(cl_int), nullptr);
    q.enqueueNDRangeKernel(
        topLevelScan,
        cl::NullRange,
        cl::NDRange(NUM_GROUPS_TOP_LEVEL_SCAN * WG_SIZE),
        cl::NDRange(WG_SIZE)
    );

    distributeSums.setArg(0, devicePartSumsMidLevel);
    distributeSums.setArg(1, devicePartSumsBottomLevel);
    distributeSums.setArg(2, (cl_uint)NUM_GROUPS_BOTTOM_LEVEL_SCAN);
    q.enqueueNDRangeKernel(
        distributeSums,
        cl::NullRange,
        cl::NDRange(NUM_GROUPS_MID_LEVEL_DISTRIBUTE * WG_SIZE),
        cl::NDRange(WG_SIZE)
    );

    distributeSums.setArg(0, devicePartSumsBottomLevel);
    distributeSums.setArg(1, output_buf);
    distributeSums.setArg(2, (cl_uint)numElems);
    q.enqueueNDRangeKernel(
        distributeSums,
        cl::NullRange,
        cl::NDRange(NUM_GROUPS_BOTTOM_LEVEL_DISTRIBUTE * WG_SIZE),
        cl::NDRange(WG_SIZE)
    );

    output.resize(input.size());
    q.enqueueReadBuffer(
        output_buf, CL_TRUE, 0, sizeof(int)*input.size(), output.data()
    );
    q.finish();
    } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return;
    }

    p.unloadCompiler();
}

std::vector<int> generate_hits(unsigned n)
{
    std::vector<int> res;
    res.reserve(n);
    for(unsigned i = 0; i < n; ++i) res.push_back(random()&1);
    return res;
}

template<typename T>
void print_vec(const std::vector<T>& t)
{
    for(unsigned i = 0; i < t.size(); ++i)
    {
        if(i != 0) std::cout << ", ";
        std::cout << t[i];
    }
    std::cout << std::endl;
}

int main()
{
    std::vector<int> hits = generate_hits(100);
    std::vector<int> indices;
    exclusive_scan_cpu(hits, indices);
    print_vec(hits);
    print_vec(indices);

    std::vector<int> cl_indices;
    exclusive_scan_cl(hits, cl_indices);

    if (indices == cl_indices) {
        std::cout << "OK: CL gave correct results" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "ERROR: CL gave wrong results" << std::endl;
        print_vec(cl_indices);
        return EXIT_FAILURE;
    }
}
