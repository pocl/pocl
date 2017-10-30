/* This test is part of pocl.
 * Author: Kalle Raiskila, 2014
 *
 * Test the OpenCL-C 'shuffle' command by looping
 * over all permutations of data and mask vector
 * lengths. Only one data type at a time is tested,
 * the type to test is passed as command line argument
 * (this allows for more interactive testsuite, and
 * non-supported types (e.g. half, double) are easier
 * to filter out).
 * The data to be shuffled is vectors where the elemnet
 * data equals the element index (i.e. [0,1,2,3]) but
 * the shuffle pattern masks are generated with rand().
 */

#include "poclu.h"
#include <cstdio>
#include <cstring>
#include <CL/cl.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

cl_context ctx;
cl_device_id did;
cl_platform_id pid;
cl_command_queue queue;

#define ERRCHECK()  if (check_cl_error(errcode, __LINE__, __FUNCTION__)) abort();

static const unsigned vecelts[] = {2,4,8,16};
static const int stimuli[] = {4, 2, 69, 4, 5, 0, 45, 16, 4, 6, 1, 18, 28, 14,
                 22, 16, 8, 2, 0, 31, 42, 11, 62, 88, 99, 23, 13};

template <typename D, typename M>
class TestShuffle {
  cl_mem mem_in1,
  mem_in2,
  mem_out,
  mem_mask1,
  mem_mask2;

  cl_program prog;

  D in1 [16] __attribute__ ((aligned (128)));
  D in2 [16] __attribute__ ((aligned (128)));
  D out [16] __attribute__ ((aligned (128)));
  M mask1 [16] __attribute__ ((aligned (128)));
  M mask2 [16] __attribute__ ((aligned (128)));
  const char* ocl_type;
  unsigned size;
  cl_int errcode;

private:
  /* Prints into std::string all the OpenCL kernel sources for each n,m combo */
  void testcase_src(std::string & src) {
    char buf[1024];
    int rv;
    unsigned n, m;
    const char* mask_type;
    switch(sizeof(M)) {
      case 1:
        mask_type = "uchar"; break;
      case 2:
        mask_type = "ushort"; break;
      case 4:
        mask_type = "uint"; break;
      case 8:
        mask_type = "ulong"; break;
      default:
        mask_type = "UNKNOWN_MASK";
    }

    for(unsigned n_loop=0; n_loop<4; n_loop++) {
        for(unsigned m_loop=0; m_loop<4; m_loop++) {

            n = vecelts[n_loop];
            m = vecelts[m_loop];
            rv = 0;
            buf[0] = 0;
            rv=sprintf(buf,
                           "__kernel void test_shuffle_%d_%d("
                           "__global %s%d *in, __global %s%d *mask, __global %s%d *out) {\n"
                           "*out = shuffle( *in, *mask);\n}\n",
                           m, n, ocl_type, m, mask_type, n, ocl_type, n);
            rv+=sprintf(buf+rv,
                           "__kernel void test_shuffle2_%d_%d("
                           "__global %s%d *in1, __global %s%d *in2, __global %s%d *mask, __global %s%d *out) {\n"
                           "*out = shuffle2( *in1, *in2, *mask);\n}\n",
                           m, n, ocl_type, m, ocl_type, m, mask_type, n, ocl_type, n);
            src.append(buf);
        }
    }
  }

  #define nsize (n==3?4:n)
  #define msize (m==3?4:m)
  // assume out is filled with 'shuffle(in, mask)'	// return true if ok
  bool output_matches_1(unsigned n, unsigned m)
  {
    bool error=false;
    for(unsigned i=0; i<n; i++)
      {
        unsigned mm = mask1[i] % msize;
        error |= (out[i] != in1[mm]);
      }
    return !error;
  }

  // assume out is filled with 'shuffle2(in1, in2, mask)'
  // return true if ok
  bool output_matches_2(unsigned n, unsigned m)
  {
    bool error=false;
    for(unsigned i=0; i<n; i++)
      {
        unsigned msk = mask2[i] % (2*msize);
        D correct = (msk < msize) ? in1[msk] : in2[msk-msize];
        if (out[i] != correct) {
            error |= true;
            printf("element %d should be %d (mask %d), got %d\n", i, (int)correct, (int)mask2[i], (int)out[i]);
          }
      }
    return !error;
  }


  // helpers: prints a vector as [0, 1, 2]
  // cast to int, so vectors of 'char' come out correctly
  void print_in1(unsigned n, unsigned m)
  {
    std::cout << "["<<(int)in1[0];
    for(unsigned i=1; i<m; i++)
      std::cout << ", " <<(int)in1[i];
    std::cout << "]";
  }
  void print_in2(unsigned n, unsigned m)
  {
    std::cout << "["<<(int)in2[0];
    for(unsigned i=1; i<m; i++)
      std::cout << ", " <<(int)in2[i];
    std::cout << "]";
  }
  void print_mask1(unsigned n, unsigned m)
  {
    std::cout << "["<<(int)mask1[0];
    for(unsigned i=1; i<n; i++)
      std::cout << ", " <<(int)mask1[i];
    std::cout << "]";
  }
  void print_mask2(unsigned n, unsigned m)
  {
    std::cout << "["<<(int)mask2[0];
    for(unsigned i=1; i<n; i++)
      std::cout << ", " <<(int)mask2[i];
    std::cout << "]";
  }
  void print_out(unsigned n, unsigned m)
  {
    std::cout << "["<<(int)out[0];
    for(unsigned i=1; i<n; i++)
      std::cout << ", " <<(int)out[i];
    std::cout << "]";
  }

  /* Run one shuffle test, return true if successful*/

  bool run_single_test(unsigned n, unsigned m){
    bool rv=true;
    cl_kernel krn, krn2;
    char kern_name[128], kern_name2[128];

    snprintf(kern_name, 128, "test_shuffle_%d_%d", m, n);
    krn = clCreateKernel(prog, kern_name, &errcode);
    ERRCHECK()

    errcode = clSetKernelArg( krn, 0, sizeof(cl_mem), &mem_in1 );
    ERRCHECK()
    errcode = clSetKernelArg( krn, 1, sizeof(cl_mem), &mem_mask1 );
    ERRCHECK()
    errcode = clSetKernelArg( krn, 2, sizeof(cl_mem), &mem_out );
    ERRCHECK()

    errcode = clEnqueueTask( queue, krn, 0, NULL, NULL );
    ERRCHECK()
    errcode = clEnqueueReadBuffer( queue, mem_out, CL_TRUE, 0, size, out, 0, NULL, NULL );
    ERRCHECK()
    errcode = clFinish(queue);
    ERRCHECK()

    if(!output_matches_1(n, m))
      {
        std::cout << "Error in shuffle " << ocl_type << " " << m;
        std::cout << " => " << ocl_type << " " << n << " :";
        print_out(n, m);
        std::cout << " = shuffle( ";
        print_in1(n, m);
        std::cout << ", ";
        print_mask1(n, m);
        std::cout << ");" << std::endl;
        rv=false;
      }

    // Now test shuffle2()
    clReleaseKernel(krn);

    snprintf(kern_name2, 128, "test_shuffle2_%d_%d", m, n);
    krn2 = clCreateKernel(prog, kern_name2, &errcode);
    ERRCHECK()
    errcode = clSetKernelArg( krn2, 0, sizeof(cl_mem), &mem_in1 );
    ERRCHECK()
    errcode = clSetKernelArg( krn2, 1, sizeof(cl_mem), &mem_in2 );
    ERRCHECK()
    errcode = clSetKernelArg( krn2, 2, sizeof(cl_mem), &mem_mask2 );
    ERRCHECK()
    errcode = clSetKernelArg( krn2, 3, sizeof(cl_mem), &mem_out );
    ERRCHECK()
    errcode = clEnqueueTask( queue, krn2, 0, NULL, NULL );
    ERRCHECK()
    errcode = clEnqueueReadBuffer( queue, mem_out, CL_TRUE, 0, size, out, 0, NULL, NULL );
    ERRCHECK()
    errcode = clFinish(queue);
    ERRCHECK()

    if(!output_matches_2(n, m))
      {
        std::cout << "Error in shuffle2 " << ocl_type << " " << m;
        std::cout << " => " << ocl_type << " " << n << " :";
        print_out(n, m);
        std::cout << " = shuffle2( ";
        print_in1(n, m);
        std::cout << ", ";
        print_in2(n, m);
        std::cout << ", ";
        print_mask2(n, m);
        std::cout << ");" << std::endl;
        rv=false;
      }
    clReleaseKernel(krn2);
    return rv;
  }





public:
  TestShuffle(const char* type) {
    ocl_type = type;
    size  = sizeof(D) * 16;
    for(unsigned i=0; i<16; i++) {
      mask1[i] = (M)stimuli[i];
      mask2[i] = (M)stimuli[i];
    }
  }

  unsigned run()
  {

    // Fixed pseudorandom stimuli to make the test deterministic.
    // Random stimuli leads to randomly appearing/disappearing
    // problems which are irritating and hard to reproduce. Values which reduce
    // to element 3 might produce an undefined value in case of 3 element inputs so
    // let's not use them in the stimulus.

    mem_in1 = clCreateBuffer(ctx,
                             CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             size, in1, &errcode);
    ERRCHECK()
    mem_in2 = clCreateBuffer(ctx,
                             CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             size, in2, &errcode);
    ERRCHECK()
    mem_mask1 = clCreateBuffer(ctx,
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              size, mask1, &errcode);
    ERRCHECK()
    mem_mask2 = clCreateBuffer(ctx,
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              size, mask2, &errcode);
    ERRCHECK()
    mem_out = clCreateBuffer(ctx,
                             CL_MEM_WRITE_ONLY,
                             size, NULL, &errcode);
    ERRCHECK()

    std::string source;
    testcase_src(source);

    const char *c_src = source.c_str();
    size_t srclen = source.size();

    prog = clCreateProgramWithSource(ctx, 1, &c_src, &srclen, &errcode);
    ERRCHECK()
    errcode = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    ERRCHECK()

    unsigned errors = 0;
    for(unsigned n_loop=0; n_loop<4; n_loop++) {
          for(unsigned m_loop=0; m_loop<4; m_loop++) {
              unsigned m = vecelts[m_loop];
              for(unsigned i=0; i<m; i++) {
                in2[i]=(D)(i+m);
                in1[i] = (D)i;
              }
              if (!run_single_test(vecelts[n_loop], vecelts[m_loop]))
                errors++;
          }
    }

    clReleaseMemObject(mem_in1);
    clReleaseMemObject(mem_in2);
    clReleaseMemObject(mem_mask1);
    clReleaseMemObject(mem_mask2);
    clReleaseMemObject(mem_out);
    clReleaseProgram(prog);

    return errors;
  }

};






int main( int argc, char *argv[])
{
	unsigned num_errors = 0;

	if( argc != 2 ) {
		std::cout << "give element type"<<std::endl;
		exit(-1);
	}

	poclu_get_any_device( &ctx, &did, &queue);

#if (__GNUC__ > 5)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

	/* Loop over input (m) and output (n) vector lengths.
	 * The big if-else is needed to pass the string
	 * representation to runtest.
	 * This cannot be fully templated, as there is a
	 * 'typedef short int half', which would cause the
	 * templating mechanism to create the test for shorts instead
	 * of halfs.
	 */
	 if( strcmp("char", argv[1]) == 0 ) {

	   TestShuffle<cl_char, cl_uchar> t("char"); num_errors = t.run();

	 } else if( strcmp("uchar", argv[1]) == 0 ) {

	   TestShuffle<cl_uchar, cl_uchar> t("uchar"); num_errors = t.run();

	 } else if( strcmp("short", argv[1]) == 0 ) {

	   TestShuffle<cl_short, cl_ushort> t("short"); num_errors = t.run();

	 } else if( strcmp("ushort", argv[1]) == 0 ) {

	   TestShuffle<cl_ushort, cl_ushort> t("ushort"); num_errors = t.run();

	 } else if( strcmp("int", argv[1]) == 0 ) {

	   TestShuffle<cl_int, cl_uint> t("int"); num_errors = t.run();

	 } else if( strcmp("uint", argv[1]) == 0 ) {

	   TestShuffle<cl_uint, cl_uint> t("uint"); num_errors = t.run();

	 } else if( strcmp("long", argv[1]) == 0 ) {

	   TestShuffle<cl_long, cl_ulong> t("long"); num_errors = t.run();

	 } else if( strcmp("ulong", argv[1]) == 0 ) {

	   TestShuffle<cl_ulong, cl_ulong> t("ulong"); num_errors = t.run();

	 } else if( strcmp("half", argv[1]) == 0 ) {

	   TestShuffle<cl_half, cl_ushort> t("half"); num_errors = t.run();

	 } else if( strcmp("float", argv[1]) == 0 ) {

	   TestShuffle<cl_float, cl_uint> t("float"); num_errors = t.run();

	 } else if( strcmp("double", argv[1]) == 0 ) {

	   TestShuffle<cl_double, cl_ulong> t("double"); num_errors = t.run();

	 } else {

	     std::cout << "Error: unknown type " << argv[1] << ": use OCL-C types"<<std::endl;
	     return -1;

	   }

#if (__GNUC__ > 5)
#pragma GCC diagnostic pop
#endif
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);

	if( num_errors == 0)
		std::cout << "OK" << std::endl;
	return num_errors;
}
