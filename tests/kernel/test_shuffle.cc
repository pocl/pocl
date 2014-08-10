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

// One shuffle testcase
// D: data type (in OCL host syntax)
// M: mask type
// n: output & mask number of vector elements
// m: in1 & in2 number of vector elements
// ocl_type: the type of D in OCL-C syntax
template <typename D, typename M>
class testcase {
public:
	unsigned n, m;
	unsigned nsize, msize;
	D *in1;
	D *in2;
	D *out;
	M *mask1;
	M *mask2;
	const char *d_type;

  testcase(unsigned n_, unsigned m_, const char* ocl_label) :
    n(n_), m(m_),
    nsize(n==3?4:n), msize(m==3?4:m),
    d_type(ocl_label) {
    // Fixed pseudorandom stimuli to make the test deterministic.
    // Random stimuli leads to randomly appearing/disappearing
    // problems which are irritating and hard to reproduce. Values which reduce
    // to element 3 might produce an undefined value in case of 3 element inputs so 
    // let's not use them in the stimulus.
    int stimuli[] = {4, 2, 69, 4, 5, 0, 45, 16, 4, 6, 1, 18, 28, 14, 
                     22, 16, 8, 2, 0, 31, 42, 11, 62, 88, 99, 23, 13};
    in1=new D[msize];       
    in2=new D[msize];
    out=new D[nsize];
    mask1=new M[nsize];
    mask2=new M[nsize];
    
    for(unsigned i=0; i<m; i++) {
      in1[i]=i;
      in2[i]=i+m;
    }
    for(unsigned i=0; i<n; i++) {
      mask1[i] = stimuli[i];
      mask2[i] = stimuli[i];
    }
  }

	int create_source(char *buf)
	{
		int rv;
		rv=sprintf(buf,
		           "__kernel void test_shuffle("
		           "__global %s%d *in, __global %s%d *mask, __global %s%d *out) {\n"
		           "*out = shuffle( *in, *mask);\n}\n",
		           d_type, m, get_mask_type(), n, d_type, n);
		rv+=sprintf(buf+rv,
		           "__kernel void test_shuffle2("
		           "__global %s%d *in1, __global %s%d *in2, __global %s%d *mask, __global %s%d *out) {\n"
		           "*out = shuffle2( *in1, *in2, *mask);\n}\n",
		           d_type, m, d_type, m, get_mask_type(), n, d_type, n);
		return rv;
	}

	// assume out is filled with 'shuffle(in, mask)'
	// return true if ok
	bool output_matches_1()
	{
		bool error=false;
		for(unsigned i=0; i<n; i++)
		{
			unsigned m = mask1[i] % msize;
			error |= out[i] != in1[m];
		}
		return !error;
	}

	// assume out is filled with 'shuffle2(in1, in2, mask)'
	// return true if ok
	bool output_matches_2()
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

	const char* get_mask_type()
	{
		switch(sizeof(M)) {
		case 1:
			return "uchar"; break;
		case 2:
			return "ushort"; break;
		case 4:
			return "uint"; break;
		case 8:
			return "ulong"; break;
		default:
			return NULL;
		}
	}

	// helpers: prints a vector as [0, 1, 2]
	// cast to int, so vectors of 'char' come out correctly
	void print_in1()
	{
		std::cout << "["<<(int)in1[0];
		for(unsigned i=1; i<m; i++)
			std::cout << ", " <<(int)in1[i];
		std::cout << "]";
	}
	void print_in2()
	{
		std::cout << "["<<(int)in2[0];
		for(unsigned i=1; i<m; i++)
			std::cout << ", " <<(int)in2[i];
		std::cout << "]";
	}
	void print_mask1()
	{
		std::cout << "["<<(int)mask1[0];
		for(unsigned i=1; i<n; i++)
			std::cout << ", " <<(int)mask1[i];
		std::cout << "]";
	}
	void print_mask2()
	{
		std::cout << "["<<(int)mask2[0];
		for(unsigned i=1; i<n; i++)
			std::cout << ", " <<(int)mask2[i];
		std::cout << "]";
	}
	void print_out()
	{
		std::cout << "["<<(int)out[0];
		for(unsigned i=1; i<n; i++)
			std::cout << ", " <<(int)out[i];
		std::cout << "]";
	}
};



/* Run one shuffle test, return true if successful*/
template<typename D, typename M>
bool runtest( int n, int m, const char* ocl_type){
	char *buf = (char*) malloc(1024);
	const char *src[1];
	src[0]=buf;
	int numchars;
	cl_mem mem_in1, mem_in2, mem_out, mem_mask1, mem_mask2;
	cl_program prog;
	cl_kernel krn;
	bool rv=true;

	testcase<D,M> tc(n, m, ocl_type);
	int size=sizeof(D);

    int mAligned = m;
    if (m == 3) mAligned = 4;

    int nAligned = n;
    if (n == 3) nAligned = 4;

	mem_in1 = clCreateBuffer(ctx,
	                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                         size * mAligned, tc.in1, NULL);
	mem_in2 = clCreateBuffer(ctx,
	                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                         size * nAligned, tc.in2, NULL);
	mem_mask1 = clCreateBuffer(ctx,
	                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                          size * nAligned, tc.mask1, NULL);
	mem_mask2 = clCreateBuffer(ctx,
	                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                          size * nAligned, tc.mask2, NULL);
	mem_out = clCreateBuffer(ctx,
	                         CL_MEM_WRITE_ONLY,
	                         size * nAligned, NULL, NULL);

	numchars = tc.create_source( buf );
	buf[numchars]=0;

	prog = clCreateProgramWithSource(ctx, 1, src, NULL, NULL);
	clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	krn = clCreateKernel(prog, "test_shuffle", NULL);

	clSetKernelArg( krn, 0, sizeof(cl_mem), &mem_in1 );
	clSetKernelArg( krn, 1, sizeof(cl_mem), &mem_mask1 );
	clSetKernelArg( krn, 2, sizeof(cl_mem), &mem_out );
	clEnqueueTask( queue, krn, 0, NULL, NULL );
	clEnqueueReadBuffer( queue, mem_out, CL_TRUE, 0, size*nAligned, tc.out, 0, NULL, NULL );
	clFinish(queue);

	if(!tc.output_matches_1()) {
		std::cout << "Error in shuffle " << ocl_type << " " << m;
		std::cout << " => " << ocl_type << " " << n << " :";
		tc.print_out();
		std::cout << " = shuffle( ";
		tc.print_in1();
		std::cout << ", ";
		tc.print_mask1();
		std::cout << ");" << std::endl;
		rv=false;
	}

	// Now test shuffle2()
	clReleaseKernel(krn);
	krn = clCreateKernel(prog, "test_shuffle2", NULL);
	clSetKernelArg( krn, 0, sizeof(cl_mem), &mem_in1 );
	clSetKernelArg( krn, 1, sizeof(cl_mem), &mem_in2 );
	clSetKernelArg( krn, 2, sizeof(cl_mem), &mem_mask2 );
	clSetKernelArg( krn, 3, sizeof(cl_mem), &mem_out );
	clEnqueueTask( queue, krn, 0, NULL, NULL );
	clEnqueueReadBuffer( queue, mem_out, CL_TRUE, 0, size*nAligned, tc.out, 0, NULL, NULL );
	clFinish(queue);

	if(!tc.output_matches_2()) {
		std::cout << "Error in shuffle2 " << ocl_type << " " << m;
		std::cout << " => " << ocl_type << " " << n << " :";
		tc.print_out();
		std::cout << " = shuffle2( ";
		tc.print_in1();
		std::cout << ", ";
		tc.print_in2();
		std::cout << ", ";
		tc.print_mask2();
		std::cout << ");" << std::endl;
		rv=false;
	}

	clReleaseMemObject(mem_in1);
	clReleaseMemObject(mem_in2);
	clReleaseMemObject(mem_mask1);
	clReleaseMemObject(mem_mask2);
	clReleaseMemObject(mem_out);
	clReleaseKernel(krn);
	clReleaseProgram(prog);

	return rv;
}

int main( int argc, char *argv[])
{
	int num_errors = 0;

	if( argc != 2 ) {
		std::cout << "give element type"<<std::endl;
		exit(-1);
	}

	poclu_get_any_device( &ctx, &did, &queue);

	/* Loop over input (m) and output (n) vector lengths.
	 * The big if-else is needed to pass the string
	 * representation to runtest.
	 * This cannot be fully templated, as there is a
	 * 'typedef short int half', which would cause the
	 * templating mechanism to create the test for shorts instead
	 * of halfs.
	 */
	int vecelts[5]={2,3,4,8,16};
	for(unsigned n_loop=0; n_loop<5; n_loop++) {
		for(unsigned m_loop=0; m_loop<5; m_loop++) {
			bool rv;
			if( strcmp("char", argv[1]) == 0 )
				rv=runtest<cl_char, cl_uchar>
				        (vecelts[n_loop], vecelts[m_loop], "char");
			else if( strcmp("uchar", argv[1]) == 0 )
				rv=runtest<cl_uchar, cl_uchar>
				        (vecelts[n_loop], vecelts[m_loop], "uchar");
			else if( strcmp("short", argv[1]) == 0 )
				rv=runtest<cl_short, cl_ushort>
				        (vecelts[n_loop], vecelts[m_loop], "short");
			else if( strcmp("ushort", argv[1]) == 0 )
				rv=runtest<cl_ushort, cl_ushort>
				        (vecelts[n_loop], vecelts[m_loop], "ushort");
			else if( strcmp("int", argv[1]) == 0 )
				rv=runtest<cl_int, cl_uint>
				        (vecelts[n_loop], vecelts[m_loop], "int");
			else if( strcmp("uint", argv[1]) == 0 )
				rv=runtest<cl_uint, cl_uint>
				        (vecelts[n_loop], vecelts[m_loop], "uint");
			else if( strcmp("long", argv[1]) == 0 )
				rv=runtest<cl_long, cl_ulong>
				        (vecelts[n_loop], vecelts[m_loop], "long");
			else if( strcmp("ulong", argv[1]) == 0 )
				rv=runtest<cl_ulong, cl_ulong>
				        (vecelts[n_loop], vecelts[m_loop], "ulong");
			else if( strcmp("half", argv[1]) == 0 )
				rv=runtest<cl_half, cl_ushort>
				        (vecelts[n_loop], vecelts[m_loop], "half");
			else if( strcmp("float", argv[1]) == 0 )
				rv=runtest<cl_float, cl_uint>
				        (vecelts[n_loop], vecelts[m_loop], "float");
			else if( strcmp("double", argv[1]) == 0 )
				rv=runtest<cl_double, cl_ulong>
				        (vecelts[n_loop], vecelts[m_loop], "double");
			else {
				std::cout << "Error: unknown type " << argv[1] << ": use OCL-C types"<<std::endl;
				return -1;
			}
			if(rv==false)
				num_errors++;
		}
	}

	if( num_errors == 0)
		std::cout << "OK" << std::endl;
	return num_errors;
}
