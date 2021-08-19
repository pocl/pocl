
// Used to cause an LLVM crash with Haswell/Broadwell.
// See https://github.com/pocl/pocl/issues/231

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iostream>

using namespace std;

const char *SOURCE = R"RAW(
#define local_barrier() barrier(CLK_LOCAL_MEM_FENCE);
#define WITHIN_KERNEL /* empty */
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_ARG __local
#define REQD_WG_SIZE(X,Y,psc_Z) __attribute__((reqd_work_group_size(X, Y, psc_Z)))
#define psc_LID_0 get_local_id(0)
#define psc_LID_1 get_local_id(1)
#define psc_LID_2 get_local_id(2)
#define psc_GID_0 get_group_id(0)
#define psc_GID_1 get_group_id(1)
#define psc_GID_2 get_group_id(2)
#define psc_LDIM_0 get_local_size(0)
#define psc_LDIM_1 get_local_size(1)
#define psc_LDIM_2 get_local_size(2)
#define psc_GDIM_0 get_num_groups(0)
#define psc_GDIM_1 get_num_groups(1)
#define psc_GDIM_2 get_num_groups(2)
    #if __OPENCL_C_VERSION__ < 120
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #endif
//CL//
#define psc_WG_SIZE 16
#define psc_SCAN_EXPR(a, b, across_seg_boundary) a+b
#define psc_INPUT_EXPR(i) (input_ary[i])
typedef int psc_scan_type;
typedef int psc_index_type;
// NO_SEG_BOUNDARY is the largest representable integer in psc_index_type.
// This assumption is used in code below.
#define NO_SEG_BOUNDARY 2147483647
//CL//
#define psc_K 256
KERNEL
REQD_WG_SIZE(psc_WG_SIZE, 1, 1)
void scan_scan_intervals_lev1(
    __global int *input_ary, __global int *output_ary,
    GLOBAL_MEM psc_scan_type *restrict psc_partial_scan_buffer,
    const psc_index_type N,
    const psc_index_type psc_interval_size
        , GLOBAL_MEM psc_scan_type *restrict psc_interval_results
    )
{
    // index psc_K in first dimension used for psc_carry storage
    struct psc_wrapped_scan_type
    {
        psc_scan_type psc_value;
    };
    // padded in psc_WG_SIZE to avoid bank conflicts
    LOCAL_MEM struct psc_wrapped_scan_type psc_ldata[psc_WG_SIZE];
    for(int i = 0; i < 10; ++i)
    {
        local_barrier();
        psc_scan_type psc_val = 0;
        if (psc_LID_0 >= 2)
        {
            psc_scan_type psc_tmp = psc_ldata[psc_LID_0 - 2].psc_value;
            psc_val = psc_tmp+ psc_val;
        }
        // {{{ writes to local allowed, reads from local not allowed
        psc_ldata[psc_LID_0].psc_value = psc_val;
    }
}
)RAW";

int main(int argc, char *argv[])
{
  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();
  cl::Program program(SOURCE, true);

#if (__GNUC__ > 5)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

  auto kernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl::Buffer>(program, "scan_scan_intervals_lev1");

  cl_int i = 0;
  cl::Buffer buffer;
  kernel(cl::EnqueueArgs(queue, cl::NDRange(16), cl::NDRange(16)),
         buffer, buffer, buffer, i, i, buffer);

#if (__GNUC__ > 5)
#pragma GCC diagnostic pop
#endif

  queue.finish();
  cl::Platform::getDefault().unloadCompiler();
}
