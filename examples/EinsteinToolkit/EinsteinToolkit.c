/* EinsteinToolkit - Evolve the Einstein equations (BSSN formulation) */

/* Note: The number of iterations can be modified. It should be
   adapted such that the total run time (reported below) is
   approximately between 10 and 100 seconds. The number of iterations
   does not influence the benchmark result, it only influences the
   benchmark's accuracy. Small numbers of iterations lead to
   inaccurate results. */
int const niters = 10;



/* Build options:
   
   Redshift, Apple's OpenCL:
   clang -I/System/Library/Frameworks/OpenCL.framework/Headers -L/System/Library/Frameworks/OpenCL.framework/Libraries -o EinsteinToolkit EinsteinToolkit.c -Wl,-framework,OpenCL
   
   Nvidia, AMD's OpenCL:
   clang -I/usr/local/AMD-APP-SDK-v2.8-RC-lnx64/include -L /usr/local/AMD-APP-SDK-v2.8-RC-lnx64/lib -o EinsteinToolkit EinsteinToolkit.c -lOpenCL -lm
   
   Nvidia, Intel's OpenCL:
   clang -I/usr/local/intel_ocl_sdk_2012_x64/usr/include -L/usr/local/intel_ocl_sdk_2012_x64/usr/lib64 -o EinsteinToolkit EinsteinToolkit.c -lOpenCL
   
   Nvidia, Nvidia's OpenCL:
   clang -I/usr/local/cuda-5.0/include -L/usr/local/cuda-5.0/lib64 -o EinsteinToolkit EinsteinToolkit.c -lOpenCL -lm
*/



/* Run times on various systems:
 *
 * Redshift, laptop, OSX, Intel(R) Core(TM) i7-3820QM CPU @ 2.70GHz:
 *    Theoretical best: 0.0393519 usec per gpu
 *    Apple's OpenCL:   0.213103  usec per gpu (with VECTOR_SIZE_I=2)
 *    pocl:             0.543815  usec per gpu (with THREAD_COUNT_ENV=4)
 *
 * Nvidia, workstation, Intel(R) Xeon(R) CPU X5675 @ 3.07GHz
 *    Theoretical best: 0.0230727 usec per gpu
 *    pocl:             0.267914  usec per gpu
*/



#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "pocl_opencl.h"

// Stringify
#define XSTR(x) #x
#define STR(x) XSTR(x)

// Divide while rounding down
static inline size_t divdown(size_t const a, size_t const b)
{
  return a/b;
}

// Divide while rounding up
static inline size_t divup(size_t const a, size_t const b)
{
  return (a+b-1)/b;
}

// Round down
static inline size_t rounddown(size_t const a, size_t const b)
{
  return divdown(a, b) * b;
}

// Round up
static inline size_t roundup(size_t const a, size_t const b)
{
  return divup(a, b) * b;
}



// Global OpenCL handles
cl_platform_id platform_id;
cl_device_id device_id;
cl_device_id main_device_id;
cl_context context;
cl_command_queue cmd_queue;
static int use_subdev;

// Code generation choices:
#define VECTORISE_ALIGNED_ARRAYS 0

// Loop traversal choices:
#define VECTOR_SIZE_I 1
#define VECTOR_SIZE_J 1
#define VECTOR_SIZE_K 1
#define UNROLL_SIZE_I 1
#define UNROLL_SIZE_J 1
#define UNROLL_SIZE_K 1
#define GROUP_SIZE_I  1
#define GROUP_SIZE_J  1
#define GROUP_SIZE_K  1
#define TILE_SIZE_I   1
#define TILE_SIZE_J   1
#define TILE_SIZE_K   1



// Cactus definitions

#define dim 3

typedef int CCTK_INT;
typedef double CCTK_REAL;



typedef struct {
  // Doubles first, then ints, to ensure proper alignment
  // Coordinates:
  double cctk_origin_space[dim];
  double cctk_delta_space[dim];
  double cctk_time;
  double cctk_delta_time;
  // Grid structure properties:
  int cctk_gsh[dim];
  int cctk_lbnd[dim];
  int cctk_lsh[dim];
  int cctk_ash[dim];
  // Loop settings:
  int lmin[dim];                 // loop region
  int lmax[dim];
  int imin[dim];                 // active region
  int imax[dim];
} cGH;



// Cactus parameters:
typedef struct {
  CCTK_REAL A_bound_limit;
  CCTK_REAL A_bound_scalar;
  CCTK_REAL A_bound_speed;
  CCTK_REAL alpha_bound_limit;
  CCTK_REAL alpha_bound_scalar;
  CCTK_REAL alpha_bound_speed;
  CCTK_REAL AlphaDriver;
  CCTK_REAL At11_bound_limit;
  CCTK_REAL At11_bound_scalar;
  CCTK_REAL At11_bound_speed;
  CCTK_REAL At12_bound_limit;
  CCTK_REAL At12_bound_scalar;
  CCTK_REAL At12_bound_speed;
  CCTK_REAL At13_bound_limit;
  CCTK_REAL At13_bound_scalar;
  CCTK_REAL At13_bound_speed;
  CCTK_REAL At22_bound_limit;
  CCTK_REAL At22_bound_scalar;
  CCTK_REAL At22_bound_speed;
  CCTK_REAL At23_bound_limit;
  CCTK_REAL At23_bound_scalar;
  CCTK_REAL At23_bound_speed;
  CCTK_REAL At33_bound_limit;
  CCTK_REAL At33_bound_scalar;
  CCTK_REAL At33_bound_speed;
  CCTK_REAL B1_bound_limit;
  CCTK_REAL B1_bound_scalar;
  CCTK_REAL B1_bound_speed;
  CCTK_REAL B2_bound_limit;
  CCTK_REAL B2_bound_scalar;
  CCTK_REAL B2_bound_speed;
  CCTK_REAL B3_bound_limit;
  CCTK_REAL B3_bound_scalar;
  CCTK_REAL B3_bound_speed;
  CCTK_REAL beta1_bound_limit;
  CCTK_REAL beta1_bound_scalar;
  CCTK_REAL beta1_bound_speed;
  CCTK_REAL beta2_bound_limit;
  CCTK_REAL beta2_bound_scalar;
  CCTK_REAL beta2_bound_speed;
  CCTK_REAL beta3_bound_limit;
  CCTK_REAL beta3_bound_scalar;
  CCTK_REAL beta3_bound_speed;
  CCTK_REAL BetaDriver;
  CCTK_REAL EpsDiss;
  CCTK_REAL gt11_bound_limit;
  CCTK_REAL gt11_bound_scalar;
  CCTK_REAL gt11_bound_speed;
  CCTK_REAL gt12_bound_limit;
  CCTK_REAL gt12_bound_scalar;
  CCTK_REAL gt12_bound_speed;
  CCTK_REAL gt13_bound_limit;
  CCTK_REAL gt13_bound_scalar;
  CCTK_REAL gt13_bound_speed;
  CCTK_REAL gt22_bound_limit;
  CCTK_REAL gt22_bound_scalar;
  CCTK_REAL gt22_bound_speed;
  CCTK_REAL gt23_bound_limit;
  CCTK_REAL gt23_bound_scalar;
  CCTK_REAL gt23_bound_speed;
  CCTK_REAL gt33_bound_limit;
  CCTK_REAL gt33_bound_scalar;
  CCTK_REAL gt33_bound_speed;
  CCTK_REAL harmonicF;
  CCTK_REAL LapseACoeff;
  CCTK_REAL LapseAdvectionCoeff;
  CCTK_REAL MinimumLapse;
  CCTK_REAL ML_curv_bound_limit;
  CCTK_REAL ML_curv_bound_scalar;
  CCTK_REAL ML_curv_bound_speed;
  CCTK_REAL ML_dtlapse_bound_limit;
  CCTK_REAL ML_dtlapse_bound_scalar;
  CCTK_REAL ML_dtlapse_bound_speed;
  CCTK_REAL ML_dtshift_bound_limit;
  CCTK_REAL ML_dtshift_bound_scalar;
  CCTK_REAL ML_dtshift_bound_speed;
  CCTK_REAL ML_Gamma_bound_limit;
  CCTK_REAL ML_Gamma_bound_scalar;
  CCTK_REAL ML_Gamma_bound_speed;
  CCTK_REAL ML_lapse_bound_limit;
  CCTK_REAL ML_lapse_bound_scalar;
  CCTK_REAL ML_lapse_bound_speed;
  CCTK_REAL ML_log_confac_bound_limit;
  CCTK_REAL ML_log_confac_bound_scalar;
  CCTK_REAL ML_log_confac_bound_speed;
  CCTK_REAL ML_metric_bound_limit;
  CCTK_REAL ML_metric_bound_scalar;
  CCTK_REAL ML_metric_bound_speed;
  CCTK_REAL ML_shift_bound_limit;
  CCTK_REAL ML_shift_bound_scalar;
  CCTK_REAL ML_shift_bound_speed;
  CCTK_REAL ML_trace_curv_bound_limit;
  CCTK_REAL ML_trace_curv_bound_scalar;
  CCTK_REAL ML_trace_curv_bound_speed;
  CCTK_REAL phi_bound_limit;
  CCTK_REAL phi_bound_scalar;
  CCTK_REAL phi_bound_speed;
  CCTK_REAL ShiftAdvectionCoeff;
  CCTK_REAL ShiftBCoeff;
  CCTK_REAL ShiftGammaCoeff;
  CCTK_REAL SpatialBetaDriverRadius;
  CCTK_REAL SpatialShiftGammaCoeffRadius;
  CCTK_REAL trK_bound_limit;
  CCTK_REAL trK_bound_scalar;
  CCTK_REAL trK_bound_speed;
  CCTK_REAL Xt1_bound_limit;
  CCTK_REAL Xt1_bound_scalar;
  CCTK_REAL Xt1_bound_speed;
  CCTK_REAL Xt2_bound_limit;
  CCTK_REAL Xt2_bound_scalar;
  CCTK_REAL Xt2_bound_speed;
  CCTK_REAL Xt3_bound_limit;
  CCTK_REAL Xt3_bound_scalar;
  CCTK_REAL Xt3_bound_speed;
  CCTK_INT conformalMethod;
  CCTK_INT fdOrder;
  CCTK_INT harmonicN;
  CCTK_INT harmonicShift;
  CCTK_INT ML_BSSN_CL_Advect_calc_every;
  CCTK_INT ML_BSSN_CL_Advect_calc_offset;
  CCTK_INT ML_BSSN_CL_boundary_calc_every;
  CCTK_INT ML_BSSN_CL_boundary_calc_offset;
  CCTK_INT ML_BSSN_CL_constraints1_calc_every;
  CCTK_INT ML_BSSN_CL_constraints1_calc_offset;
  CCTK_INT ML_BSSN_CL_constraints2_calc_every;
  CCTK_INT ML_BSSN_CL_constraints2_calc_offset;
  CCTK_INT ML_BSSN_CL_convertFromADMBase_calc_every;
  CCTK_INT ML_BSSN_CL_convertFromADMBase_calc_offset;
  CCTK_INT ML_BSSN_CL_convertFromADMBaseGamma_calc_every;
  CCTK_INT ML_BSSN_CL_convertFromADMBaseGamma_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBase_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBase_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_offset;
  CCTK_INT ML_BSSN_CL_Dissipation_calc_every;
  CCTK_INT ML_BSSN_CL_Dissipation_calc_offset;
  CCTK_INT ML_BSSN_CL_enforce_calc_every;
  CCTK_INT ML_BSSN_CL_enforce_calc_offset;
  CCTK_INT ML_BSSN_CL_InitGamma_calc_every;
  CCTK_INT ML_BSSN_CL_InitGamma_calc_offset;
  CCTK_INT ML_BSSN_CL_InitRHS_calc_every;
  CCTK_INT ML_BSSN_CL_InitRHS_calc_offset;
  CCTK_INT ML_BSSN_CL_MaxNumArrayEvolvedVars;
  CCTK_INT ML_BSSN_CL_MaxNumEvolvedVars;
  CCTK_INT ML_BSSN_CL_Minkowski_calc_every;
  CCTK_INT ML_BSSN_CL_Minkowski_calc_offset;
  CCTK_INT ML_BSSN_CL_RHS1_calc_every;
  CCTK_INT ML_BSSN_CL_RHS1_calc_offset;
  CCTK_INT ML_BSSN_CL_RHS2_calc_every;
  CCTK_INT ML_BSSN_CL_RHS2_calc_offset;
  CCTK_INT ML_BSSN_CL_RHSStaticBoundary_calc_every;
  CCTK_INT ML_BSSN_CL_RHSStaticBoundary_calc_offset;
  CCTK_INT other_timelevels;
  CCTK_INT rhs_timelevels;
  CCTK_INT ShiftAlphaPower;
  CCTK_INT timelevels;
  CCTK_INT verbose;
} cctk_parameters_t;



typedef struct {
  CCTK_REAL* ptr;
  cl_mem mem;
} ptr_t;

typedef struct {
  ptr_t x;
  ptr_t y;
  ptr_t z;
  ptr_t r;
  ptr_t At11;
  ptr_t At11_p;
  ptr_t At11_p_p;
  ptr_t At12;
  ptr_t At12_p;
  ptr_t At12_p_p;
  ptr_t At13;
  ptr_t At13_p;
  ptr_t At13_p_p;
  ptr_t At22;
  ptr_t At22_p;
  ptr_t At22_p_p;
  ptr_t At23;
  ptr_t At23_p;
  ptr_t At23_p_p;
  ptr_t At33;
  ptr_t At33_p;
  ptr_t At33_p_p;
  ptr_t A;
  ptr_t A_p;
  ptr_t A_p_p;
  ptr_t Arhs;
  ptr_t B1;
  ptr_t B1_p;
  ptr_t B1_p_p;
  ptr_t B2;
  ptr_t B2_p;
  ptr_t B2_p_p;
  ptr_t B3;
  ptr_t B3_p;
  ptr_t B3_p_p;
  ptr_t B1rhs;
  ptr_t B2rhs;
  ptr_t B3rhs;
  ptr_t Xt1;
  ptr_t Xt1_p;
  ptr_t Xt1_p_p;
  ptr_t Xt2;
  ptr_t Xt2_p;
  ptr_t Xt2_p_p;
  ptr_t Xt3;
  ptr_t Xt3_p;
  ptr_t Xt3_p_p;
  ptr_t Xt1rhs;
  ptr_t Xt2rhs;
  ptr_t Xt3rhs;
  ptr_t alpha;
  ptr_t alpha_p;
  ptr_t alpha_p_p;
  ptr_t alpharhs;
  ptr_t phi;
  ptr_t phi_p;
  ptr_t phi_p_p;
  ptr_t phirhs;
  ptr_t gt11;
  ptr_t gt11_p;
  ptr_t gt11_p_p;
  ptr_t gt12;
  ptr_t gt12_p;
  ptr_t gt12_p_p;
  ptr_t gt13;
  ptr_t gt13_p;
  ptr_t gt13_p_p;
  ptr_t gt22;
  ptr_t gt22_p;
  ptr_t gt22_p_p;
  ptr_t gt23;
  ptr_t gt23_p;
  ptr_t gt23_p_p;
  ptr_t gt33;
  ptr_t gt33_p;
  ptr_t gt33_p_p;
  ptr_t gt11rhs;
  ptr_t gt12rhs;
  ptr_t gt13rhs;
  ptr_t gt22rhs;
  ptr_t gt23rhs;
  ptr_t gt33rhs;
  ptr_t beta1;
  ptr_t beta1_p;
  ptr_t beta1_p_p;
  ptr_t beta2;
  ptr_t beta2_p;
  ptr_t beta2_p_p;
  ptr_t beta3;
  ptr_t beta3_p;
  ptr_t beta3_p_p;
  ptr_t beta1rhs;
  ptr_t beta2rhs;
  ptr_t beta3rhs;
  ptr_t trK;
  ptr_t trK_p;
  ptr_t trK_p_p;
  ptr_t trKrhs;
  ptr_t At11rhs;
  ptr_t At12rhs;
  ptr_t At13rhs;
  ptr_t At22rhs;
  ptr_t At23rhs;
  ptr_t At33rhs;
} cctk_arguments_t;



static void allocate(cGH const* const cctkGH,
                     ptr_t* const ptr,
                     CCTK_REAL const val)
{
  int const nsize =
    cctkGH->cctk_ash[0] * cctkGH->cctk_ash[1] * cctkGH->cctk_ash[2];
  ptr->ptr = malloc(nsize * sizeof(CCTK_REAL));
  assert(ptr->ptr);
  for (int k=0; k<cctkGH->cctk_lsh[2]; ++k) {
    for (int j=0; j<cctkGH->cctk_lsh[1]; ++j) {
      for (int i=0; i<cctkGH->cctk_lsh[0]; ++i) {
        int const ind3d =
          i + cctkGH->cctk_ash[0] * (j + cctkGH->cctk_ash[1] * k);
        ptr->ptr[ind3d] = val;
      }
    }
  }
  ptr->mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                            nsize * sizeof(CCTK_REAL), ptr->ptr, NULL);
  assert(ptr->mem);
}

static void deallocate(cGH const* const cctkGH,
                     ptr_t* const ptr,
                     CCTK_REAL const val)
{
  assert(ptr->ptr);
  clReleaseMemObject(ptr->mem);
  free(ptr->ptr);
}

static cl_mem mem_cctkGH;
static cl_mem mem_cctk_parameters;

static cl_program program1;
static cl_kernel kernel1;

static cl_program program2;
static cl_kernel kernel2;

void setup(const char* program_source1, const char* program_source2)
{
  cl_int cerr;
  
  // Choose a platform and a context (basically a device)
  cl_uint num_platforms;
  clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id platform_ids[num_platforms];
  clGetPlatformIDs(num_platforms, &platform_ids[0], &num_platforms);
  if (num_platforms <= 0) {
    fprintf(stderr, "No OpenCL platforms found\n");
    assert(0);
  }
  assert(num_platforms > 0);
  
  cl_device_type const want_device_types = CL_DEVICE_TYPE_CPU;
  // CL_DEVICE_TYPE_GPU
  // CL_DEVICE_TYPE_ACCELERATOR
  // CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
  
  // Loop over all platforms
  platform_id = 0;
  for (cl_uint platform = 0; platform < num_platforms; ++platform) {
    
    cl_platform_id const tmp_platform_id = platform_ids[platform];
    printf("OpenCL platform #%d:\n", platform);
    
    size_t platform_name_length;
    clGetPlatformInfo(tmp_platform_id, CL_PLATFORM_NAME,
                      0, NULL, &platform_name_length);
    char platform_name[platform_name_length];
    clGetPlatformInfo(tmp_platform_id, CL_PLATFORM_NAME,
                      platform_name_length, platform_name, NULL);
    printf("   OpenCL platform name: %s\n", platform_name);
    size_t platform_vendor_length;
    clGetPlatformInfo(tmp_platform_id, CL_PLATFORM_VENDOR,
                      0, NULL, &platform_vendor_length);
    char platform_vendor[platform_vendor_length];
    clGetPlatformInfo(tmp_platform_id, CL_PLATFORM_VENDOR,
                      platform_vendor_length, platform_vendor, NULL);
    printf("   OpenCL platform vendor: %s\n", platform_vendor);
    
    cl_context_properties const cprops[] =
      {CL_CONTEXT_PLATFORM, (cl_context_properties)tmp_platform_id, 0};
    context =
      clCreateContextFromType(cprops, want_device_types, NULL, NULL, &cerr);
    if (cerr == CL_SUCCESS) {
      platform_id = tmp_platform_id;
    }
  }
  if (platform_id == 0) {
    // Could not find a context on any platform, abort
    fprintf(stderr, "Could not create OpenCL context for selected device type\n");
    assert(0);
  }
  
  size_t ndevice_ids;
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &ndevice_ids);
  ndevice_ids /= sizeof(cl_device_id);
  cl_device_id device_ids[ndevice_ids];
  clGetContextInfo(context, CL_CONTEXT_DEVICES,
                   ndevice_ids*sizeof(cl_device_id), device_ids, NULL);
  assert(ndevice_ids >= 1);
  main_device_id = device_ids[0];

  if (use_subdev)
    {
      {
        cl_uint max_cus;
        int err = clGetDeviceInfo (main_device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                                   sizeof (max_cus), &max_cus, NULL);
        assert (err == CL_SUCCESS);
        if (max_cus < 2)
          {
            fprintf (stderr,
                     "Insufficient compute units for subdevice creation\n");
            exit (77);
          }
      }
      const cl_device_partition_property props[]
          = { CL_DEVICE_PARTITION_EQUALLY, 2, 0 };
      cl_device_id subdevs[128];
      cl_uint retval;
      int err
          = clCreateSubDevices (main_device_id, props, 128, subdevs, &retval);
      assert (err == CL_SUCCESS);
      device_id = subdevs[0];
    }
  else
    device_id = main_device_id;

  size_t device_name_length;
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &device_name_length);
  char device_name[device_name_length];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME,
                  device_name_length, device_name, NULL);
  printf("OpenCL device name: %s\n", device_name);
  
  clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM,
                  sizeof platform_id, &platform_id, NULL);
  size_t platform_name_length;
  clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
                    0, NULL, &platform_name_length);
  char platform_name[platform_name_length];
  clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
                    platform_name_length, platform_name, NULL);
  printf("OpenCL platform name: %s\n", platform_name);
  size_t platform_vendor_length;
  clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR,
                    0, NULL, &platform_vendor_length);
  char platform_vendor[platform_vendor_length];
  clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR,
                    platform_vendor_length, platform_vendor, NULL);
  printf("OpenCL platform vendor: %s\n", platform_vendor);
  
  cmd_queue = clCreateCommandQueue(context, device_id, 0, NULL);
  assert(cmd_queue);


  char const* const options =
    "-DVECTORISE_ALIGNED_ARRAYS=" STR(VECTORISE_ALIGNED_ARRAYS) " "
    "-DVECTOR_SIZE_I=" STR(VECTOR_SIZE_I) " "
    "-DVECTOR_SIZE_J=" STR(VECTOR_SIZE_J) " "
    "-DVECTOR_SIZE_K=" STR(VECTOR_SIZE_K) " "
    "-DUNROLL_SIZE_I=" STR(UNROLL_SIZE_I) " "
    "-DUNROLL_SIZE_J=" STR(UNROLL_SIZE_J) " "
    "-DUNROLL_SIZE_K=" STR(UNROLL_SIZE_K) " "
    "-DGROUP_SIZE_I=" STR(GROUP_SIZE_I) " "
    "-DGROUP_SIZE_J=" STR(GROUP_SIZE_J) " "
    "-DGROUP_SIZE_K=" STR(GROUP_SIZE_K) " "
    "-DTILE_SIZE_I=" STR(TILE_SIZE_I) " "
    "-DTILE_SIZE_J=" STR(TILE_SIZE_J) " "
    "-DTILE_SIZE_K=" STR(TILE_SIZE_K) " ";

  int ierr;

  program1 =
    clCreateProgramWithSource(context, 1, (const char**)&program_source1,
                              NULL, NULL);
  assert(program1);

  ierr = clBuildProgram(program1, 0, NULL, options, NULL, NULL);
  if (ierr) {
    size_t log_size;
    ierr = clGetProgramBuildInfo(program1, device_id,
                                 CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    assert(!ierr);
    char build_log[log_size];
    ierr = clGetProgramBuildInfo(program1, device_id,
                                 CL_PROGRAM_BUILD_LOG,
                                 log_size, build_log, NULL);
    assert(!ierr);
    printf("Build log:\n"
           "********************************************************************************\n"
           "%s\n"
           "********************************************************************************\n", build_log);
    assert(0);
  }

  kernel1 = clCreateKernel(program1, "ML_BSSN_CL_RHS1", NULL);
  assert(kernel1);

  program2 =
    clCreateProgramWithSource(context, 1, (const char**)&program_source2,
                              NULL, NULL);
  assert(program2);

  ierr = clBuildProgram(program2, 0, NULL, options, NULL, NULL);
  assert(!ierr);

  kernel2 = clCreateKernel(program2, "ML_BSSN_CL_RHS2", NULL);
  assert(kernel2);

}

void cleanup() {

  clReleaseKernel(kernel1);
  clReleaseProgram(program1);

  clReleaseKernel(kernel2);
  clReleaseProgram(program2);

  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);
  clUnloadPlatformCompiler (platform_id);
}

void init(cGH              * const cctkGH,
          cctk_parameters_t* const cctk_parameters,
          cctk_arguments_t * const cctk_arguments)
{
  cctkGH->cctk_origin_space[0] = 0.0;
  cctkGH->cctk_origin_space[1] = 0.0;
  cctkGH->cctk_origin_space[2] = 0.0;
  cctkGH->cctk_delta_space[0] = 1.0;
  cctkGH->cctk_delta_space[1] = 1.0;
  cctkGH->cctk_delta_space[2] = 1.0;
  cctkGH->cctk_time = 0.0;
  cctkGH->cctk_delta_time = 1.0;
  cctkGH->cctk_gsh[0] = 70;
  cctkGH->cctk_gsh[1] = 70;
  cctkGH->cctk_gsh[2] = 70;
  cctkGH->cctk_lbnd[0] = 0;
  cctkGH->cctk_lbnd[1] = 0;
  cctkGH->cctk_lbnd[2] = 0;
  cctkGH->cctk_lsh[0] = cctkGH->cctk_gsh[0];
  cctkGH->cctk_lsh[1] = cctkGH->cctk_gsh[1];
  cctkGH->cctk_lsh[2] = cctkGH->cctk_gsh[2];
  cctkGH->cctk_ash[0] = roundup(cctkGH->cctk_lsh[0], VECTOR_SIZE_I);
  cctkGH->cctk_ash[1] = roundup(cctkGH->cctk_lsh[1], VECTOR_SIZE_J);
  cctkGH->cctk_ash[2] = roundup(cctkGH->cctk_lsh[2], VECTOR_SIZE_K);
  // Looping region (for all threads combined)
  cctkGH->imin[0] = 3;
  cctkGH->imin[1] = 3;
  cctkGH->imin[2] = 3;
  cctkGH->imax[0] = cctkGH->cctk_lsh[0] - 3;
  cctkGH->imax[1] = cctkGH->cctk_lsh[1] - 3;
  cctkGH->imax[2] = cctkGH->cctk_lsh[2] - 3;
  // Active region (for this thread)
  cctkGH->lmin[0] = rounddown(cctkGH->imin[0], VECTOR_SIZE_I);
  cctkGH->lmin[1] = rounddown(cctkGH->imin[1], VECTOR_SIZE_J);
  cctkGH->lmin[2] = rounddown(cctkGH->imin[2], VECTOR_SIZE_K);
  cctkGH->lmax[0] = cctkGH->lmin[0] + roundup(cctkGH->imax[0] - cctkGH->lmin[0],
                                              VECTOR_SIZE_I * UNROLL_SIZE_I);
  cctkGH->lmax[1] = cctkGH->lmin[1] + roundup(cctkGH->imax[1] - cctkGH->lmin[1],
                                              VECTOR_SIZE_J * UNROLL_SIZE_J);
  cctkGH->lmax[2] = cctkGH->lmin[2] + roundup(cctkGH->imax[2] - cctkGH->lmin[2],
                                              VECTOR_SIZE_K * UNROLL_SIZE_K);
  printf("cctkGH:\n");
  printf("   gsh=[%d,%d,%d]\n", cctkGH->cctk_gsh[0], cctkGH->cctk_gsh[1], cctkGH->cctk_gsh[2]);
  printf("   lbnd=[%d,%d,%d]\n", cctkGH->cctk_lbnd[0], cctkGH->cctk_lbnd[1], cctkGH->cctk_lbnd[2]);
  printf("   lsh=[%d,%d,%d]\n", cctkGH->cctk_lsh[0], cctkGH->cctk_lsh[1], cctkGH->cctk_lsh[2]);
  printf("   ash=[%d,%d,%d]\n", cctkGH->cctk_ash[0], cctkGH->cctk_ash[1], cctkGH->cctk_ash[2]);
  printf("   imin=[%d,%d,%d]\n", cctkGH->imin[0], cctkGH->imin[1], cctkGH->imin[2]);
  printf("   imax=[%d,%d,%d]\n", cctkGH->imax[0], cctkGH->imax[1], cctkGH->imax[2]);
  printf("   lmin=[%d,%d,%d]\n", cctkGH->lmin[0], cctkGH->lmin[1], cctkGH->lmin[2]);
  printf("   lmax=[%d,%d,%d]\n", cctkGH->lmax[0], cctkGH->lmax[1], cctkGH->lmax[2]);
  
  /* cctk_parameters->A_bound_limit = 0.0; */
  /* cctk_parameters->A_bound_scalar = 0.0; */
  /* cctk_parameters->A_bound_speed = 0.0; */
  /* cctk_parameters->alpha_bound_limit = 0.0; */
  /* cctk_parameters->alpha_bound_scalar = 0.0; */
  /* cctk_parameters->alpha_bound_speed = 0.0; */
  /* cctk_parameters->AlphaDriver = 1.0; */
  /* cctk_parameters->At11_bound_limit = 0.0; */
  /* cctk_parameters->At11_bound_scalar = 0.0; */
  /* cctk_parameters->At11_bound_speed = 0.0; */
  /* cctk_parameters->At12_bound_limit = 0.0; */
  /* cctk_parameters->At12_bound_scalar = 0.0; */
  /* cctk_parameters->At12_bound_speed = 0.0; */
  /* cctk_parameters->At13_bound_limit = 0.0; */
  /* cctk_parameters->At13_bound_scalar = 0.0; */
  /* cctk_parameters->At13_bound_speed = 0.0; */
  /* cctk_parameters->At22_bound_limit = 0.0; */
  /* cctk_parameters->At22_bound_scalar = 0.0; */
  /* cctk_parameters->At22_bound_speed = 0.0; */
  /* cctk_parameters->At23_bound_limit = 0.0; */
  /* cctk_parameters->At23_bound_scalar = 0.0; */
  /* cctk_parameters->At23_bound_speed = 0.0; */
  /* cctk_parameters->At33_bound_limit = 0.0; */
  /* cctk_parameters->At33_bound_scalar = 0.0; */
  /* cctk_parameters->At33_bound_speed = 0.0; */
  /* cctk_parameters->B1_bound_limit = 0.0; */
  /* cctk_parameters->B1_bound_scalar = 0.0; */
  /* cctk_parameters->B1_bound_speed = 0.0; */
  /* cctk_parameters->B2_bound_limit = 0.0; */
  /* cctk_parameters->B2_bound_scalar = 0.0; */
  /* cctk_parameters->B2_bound_speed = 0.0; */
  /* cctk_parameters->B3_bound_limit = 0.0; */
  /* cctk_parameters->B3_bound_scalar = 0.0; */
  /* cctk_parameters->B3_bound_speed = 0.0; */
  /* cctk_parameters->beta1_bound_limit = 0.0; */
  /* cctk_parameters->beta1_bound_scalar = 0.0; */
  /* cctk_parameters->beta1_bound_speed = 0.0; */
  /* cctk_parameters->beta2_bound_limit = 0.0; */
  /* cctk_parameters->beta2_bound_scalar = 0.0; */
  /* cctk_parameters->beta2_bound_speed = 0.0; */
  /* cctk_parameters->beta3_bound_limit = 0.0; */
  /* cctk_parameters->beta3_bound_scalar = 0.0; */
  /* cctk_parameters->beta3_bound_speed = 0.0; */
  /* cctk_parameters->BetaDriver = 1.0; */
  /* cctk_parameters->EpsDiss = 0.2; */
  /* cctk_parameters->gt11_bound_limit = 0.0; */
  /* cctk_parameters->gt11_bound_scalar = 0.0; */
  /* cctk_parameters->gt11_bound_speed = 0.0; */
  /* cctk_parameters->gt12_bound_limit = 0.0; */
  /* cctk_parameters->gt12_bound_scalar = 0.0; */
  /* cctk_parameters->gt12_bound_speed = 0.0; */
  /* cctk_parameters->gt13_bound_limit = 0.0; */
  /* cctk_parameters->gt13_bound_scalar = 0.0; */
  /* cctk_parameters->gt13_bound_speed = 0.0; */
  /* cctk_parameters->gt22_bound_limit = 0.0; */
  /* cctk_parameters->gt22_bound_scalar = 0.0; */
  /* cctk_parameters->gt22_bound_speed = 0.0; */
  /* cctk_parameters->gt23_bound_limit = 0.0; */
  /* cctk_parameters->gt23_bound_scalar = 0.0; */
  /* cctk_parameters->gt23_bound_speed = 0.0; */
  /* cctk_parameters->gt33_bound_limit = 0.0; */
  /* cctk_parameters->gt33_bound_scalar = 0.0; */
  /* cctk_parameters->gt33_bound_speed = 0.0; */
  /* cctk_parameters->harmonicF = 2.0; */
  /* cctk_parameters->LapseACoeff = 1.0; */
  /* cctk_parameters->LapseAdvectionCoeff = 1.0; */
  /* cctk_parameters->MinimumLapse = 0.0; */
  /* cctk_parameters->ML_curv_bound_limit = 0.0; */
  /* cctk_parameters->ML_curv_bound_scalar = 0.0; */
  /* cctk_parameters->ML_curv_bound_speed = 0.0; */
  /* cctk_parameters->ML_dtlapse_bound_limit = 0.0; */
  /* cctk_parameters->ML_dtlapse_bound_scalar = 0.0; */
  /* cctk_parameters->ML_dtlapse_bound_speed = 0.0; */
  /* cctk_parameters->ML_dtshift_bound_limit = 0.0; */
  /* cctk_parameters->ML_dtshift_bound_scalar = 0.0; */
  /* cctk_parameters->ML_dtshift_bound_speed = 0.0; */
  /* cctk_parameters->ML_Gamma_bound_limit = 0.0; */
  /* cctk_parameters->ML_Gamma_bound_scalar = 0.0; */
  /* cctk_parameters->ML_Gamma_bound_speed = 0.0; */
  /* cctk_parameters->ML_lapse_bound_limit = 0.0; */
  /* cctk_parameters->ML_lapse_bound_scalar = 0.0; */
  /* cctk_parameters->ML_lapse_bound_speed = 0.0; */
  /* cctk_parameters->ML_log_confac_bound_limit = 0.0; */
  /* cctk_parameters->ML_log_confac_bound_scalar = 0.0; */
  /* cctk_parameters->ML_log_confac_bound_speed = 0.0; */
  /* cctk_parameters->ML_metric_bound_limit = 0.0; */
  /* cctk_parameters->ML_metric_bound_scalar = 0.0; */
  /* cctk_parameters->ML_metric_bound_speed = 0.0; */
  /* cctk_parameters->ML_shift_bound_limit = 0.0; */
  /* cctk_parameters->ML_shift_bound_scalar = 0.0; */
  /* cctk_parameters->ML_shift_bound_speed = 0.0; */
  /* cctk_parameters->ML_trace_curv_bound_limit = 0.0; */
  /* cctk_parameters->ML_trace_curv_bound_scalar = 0.0; */
  /* cctk_parameters->ML_trace_curv_bound_speed = 0.0; */
  /* cctk_parameters->phi_bound_limit = 0.0; */
  /* cctk_parameters->phi_bound_scalar = 0.0; */
  /* cctk_parameters->phi_bound_speed = 0.0; */
  /* cctk_parameters->ShiftAdvectionCoeff = 1.0; */
  /* cctk_parameters->ShiftBCoeff = 1.0; */
  /* cctk_parameters->ShiftGammaCoeff = 0.75; */
  /* cctk_parameters->SpatialBetaDriverRadius = 1.0e+10; */
  /* cctk_parameters->SpatialShiftGammaCoeffRadius = 1.0e+10; */
  /* cctk_parameters->trK_bound_limit = 0.0; */
  /* cctk_parameters->trK_bound_scalar = 0.0; */
  /* cctk_parameters->trK_bound_speed = 0.0; */
  /* cctk_parameters->Xt1_bound_limit = 0.0; */
  /* cctk_parameters->Xt1_bound_scalar = 0.0; */
  /* cctk_parameters->Xt1_bound_speed = 0.0; */
  /* cctk_parameters->Xt2_bound_limit = 0.0; */
  /* cctk_parameters->Xt2_bound_scalar = 0.0; */
  /* cctk_parameters->Xt2_bound_speed = 0.0; */
  /* cctk_parameters->Xt3_bound_limit = 0.0; */
  /* cctk_parameters->Xt3_bound_scalar = 0.0; */
  /* cctk_parameters->Xt3_bound_speed = 0.0; */
  /* cctk_parameters->conformalMethod = 0; */
  /* cctk_parameters->fdOrder = 4; */
  /* cctk_parameters->harmonicN = 1; */
  /* cctk_parameters->harmonicShift = 0; */
  /* cctk_parameters->ML_BSSN_CL_Advect_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_Advect_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_boundary_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_boundary_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_constraints1_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_constraints1_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_constraints2_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_constraints2_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertFromADMBase_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertFromADMBase_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertFromADMBaseGamma_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertFromADMBaseGamma_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBase_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBase_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_Dissipation_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_Dissipation_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_enforce_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_enforce_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_InitGamma_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_InitGamma_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_InitRHS_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_InitRHS_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_MaxNumArrayEvolvedVars = 0; */
  /* cctk_parameters->ML_BSSN_CL_MaxNumEvolvedVars = 0; */
  /* cctk_parameters->ML_BSSN_CL_Minkowski_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_Minkowski_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_RHS1_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_RHS1_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_RHS2_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_RHS2_calc_offset = 0; */
  /* cctk_parameters->ML_BSSN_CL_RHSStaticBoundary_calc_every = 0; */
  /* cctk_parameters->ML_BSSN_CL_RHSStaticBoundary_calc_offset = 0; */
  /* cctk_parameters->other_timelevels = 1; */
  /* cctk_parameters->rhs_timelevels = 1; */
  /* cctk_parameters->ShiftAlphaPower = 0; */
  /* cctk_parameters->timelevels = 3; */
  /* cctk_parameters->verbose = 0; */
  
  allocate(cctkGH, &cctk_arguments->x, 10.0);
  allocate(cctkGH, &cctk_arguments->y, 11.0);
  allocate(cctkGH, &cctk_arguments->z, 12.0);
  allocate(cctkGH, &cctk_arguments->r, 13.0);
  allocate(cctkGH, &cctk_arguments->At11, 0.0);
  allocate(cctkGH, &cctk_arguments->At11_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At11_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At12, 0.0);
  allocate(cctkGH, &cctk_arguments->At12_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At12_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At13, 0.0);
  allocate(cctkGH, &cctk_arguments->At13_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At13_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At22, 0.0);
  allocate(cctkGH, &cctk_arguments->At22_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At22_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At23, 0.0);
  allocate(cctkGH, &cctk_arguments->At23_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At23_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At33, 0.0);
  allocate(cctkGH, &cctk_arguments->At33_p, 0.0);
  allocate(cctkGH, &cctk_arguments->At33_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->A, 0.0);
  allocate(cctkGH, &cctk_arguments->A_p, 0.0);
  allocate(cctkGH, &cctk_arguments->A_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Arhs, -1.0);
  allocate(cctkGH, &cctk_arguments->B1, 0.0);
  allocate(cctkGH, &cctk_arguments->B1_p, 0.0);
  allocate(cctkGH, &cctk_arguments->B1_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->B2, 0.0);
  allocate(cctkGH, &cctk_arguments->B2_p, 0.0);
  allocate(cctkGH, &cctk_arguments->B2_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->B3, 0.0);
  allocate(cctkGH, &cctk_arguments->B3_p, 0.0);
  allocate(cctkGH, &cctk_arguments->B3_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->B1rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->B2rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->B3rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->Xt1, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt1_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt1_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt2, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt2_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt2_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt3, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt3_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt3_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->Xt1rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->Xt2rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->Xt3rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->alpha, 1.0);
  allocate(cctkGH, &cctk_arguments->alpha_p, 1.0);
  allocate(cctkGH, &cctk_arguments->alpha_p_p, 1.0);
  allocate(cctkGH, &cctk_arguments->alpharhs, -1.0);
  allocate(cctkGH, &cctk_arguments->phi, 0.0);
  allocate(cctkGH, &cctk_arguments->phi_p, 0.0);
  allocate(cctkGH, &cctk_arguments->phi_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->phirhs, -1.0);
  allocate(cctkGH, &cctk_arguments->gt11, 1.0);
  allocate(cctkGH, &cctk_arguments->gt11_p, 1.0);
  allocate(cctkGH, &cctk_arguments->gt11_p_p, 1.0);
  allocate(cctkGH, &cctk_arguments->gt12, 0.0);
  allocate(cctkGH, &cctk_arguments->gt12_p, 0.0);
  allocate(cctkGH, &cctk_arguments->gt12_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->gt13, 0.0);
  allocate(cctkGH, &cctk_arguments->gt13_p, 0.0);
  allocate(cctkGH, &cctk_arguments->gt13_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->gt22, 1.0);
  allocate(cctkGH, &cctk_arguments->gt22_p, 1.0);
  allocate(cctkGH, &cctk_arguments->gt22_p_p, 1.0);
  allocate(cctkGH, &cctk_arguments->gt23, 0.0);
  allocate(cctkGH, &cctk_arguments->gt23_p, 0.0);
  allocate(cctkGH, &cctk_arguments->gt23_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->gt33, 1.0);
  allocate(cctkGH, &cctk_arguments->gt33_p, 1.0);
  allocate(cctkGH, &cctk_arguments->gt33_p_p, 1.0);
  allocate(cctkGH, &cctk_arguments->gt11rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->gt12rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->gt13rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->gt22rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->gt23rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->gt33rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->beta1, 0.0);
  allocate(cctkGH, &cctk_arguments->beta1_p, 0.0);
  allocate(cctkGH, &cctk_arguments->beta1_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->beta2, 0.0);
  allocate(cctkGH, &cctk_arguments->beta2_p, 0.0);
  allocate(cctkGH, &cctk_arguments->beta2_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->beta3, 0.0);
  allocate(cctkGH, &cctk_arguments->beta3_p, 0.0);
  allocate(cctkGH, &cctk_arguments->beta3_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->beta1rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->beta2rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->beta3rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->trK, 0.0);
  allocate(cctkGH, &cctk_arguments->trK_p, 0.0);
  allocate(cctkGH, &cctk_arguments->trK_p_p, 0.0);
  allocate(cctkGH, &cctk_arguments->trKrhs, -1.0);
  allocate(cctkGH, &cctk_arguments->At11rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->At12rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->At13rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->At22rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->At23rhs, -1.0);
  allocate(cctkGH, &cctk_arguments->At33rhs, -1.0);

  mem_cctkGH =
    clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                   sizeof *cctkGH, (cGH*)cctkGH, NULL);
  assert(mem_cctkGH);

  mem_cctk_parameters =
    clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                   sizeof *cctk_parameters, (cctk_parameters_t*)cctk_parameters, NULL);
  assert(mem_cctk_parameters);

}

void deinit(cGH              * const cctkGH,
          cctk_parameters_t* const cctk_parameters,
          cctk_arguments_t * const cctk_arguments)
{

  clReleaseMemObject(mem_cctkGH);
  clReleaseMemObject(mem_cctk_parameters);

  deallocate(cctkGH, &cctk_arguments->x, 10.0);
  deallocate(cctkGH, &cctk_arguments->y, 11.0);
  deallocate(cctkGH, &cctk_arguments->z, 12.0);
  deallocate(cctkGH, &cctk_arguments->r, 13.0);
  deallocate(cctkGH, &cctk_arguments->At11, 0.0);
  deallocate(cctkGH, &cctk_arguments->At11_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At11_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At12, 0.0);
  deallocate(cctkGH, &cctk_arguments->At12_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At12_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At13, 0.0);
  deallocate(cctkGH, &cctk_arguments->At13_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At13_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At22, 0.0);
  deallocate(cctkGH, &cctk_arguments->At22_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At22_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At23, 0.0);
  deallocate(cctkGH, &cctk_arguments->At23_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At23_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At33, 0.0);
  deallocate(cctkGH, &cctk_arguments->At33_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->At33_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->A, 0.0);
  deallocate(cctkGH, &cctk_arguments->A_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->A_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Arhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->B1, 0.0);
  deallocate(cctkGH, &cctk_arguments->B1_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->B1_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->B2, 0.0);
  deallocate(cctkGH, &cctk_arguments->B2_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->B2_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->B3, 0.0);
  deallocate(cctkGH, &cctk_arguments->B3_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->B3_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->B1rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->B2rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->B3rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->Xt1, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt1_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt1_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt2, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt2_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt2_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt3, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt3_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt3_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->Xt1rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->Xt2rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->Xt3rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->alpha, 1.0);
  deallocate(cctkGH, &cctk_arguments->alpha_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->alpha_p_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->alpharhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->phi, 0.0);
  deallocate(cctkGH, &cctk_arguments->phi_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->phi_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->phirhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->gt11, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt11_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt11_p_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt12, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt12_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt12_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt13, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt13_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt13_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt22, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt22_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt22_p_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt23, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt23_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt23_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->gt33, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt33_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt33_p_p, 1.0);
  deallocate(cctkGH, &cctk_arguments->gt11rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->gt12rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->gt13rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->gt22rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->gt23rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->gt33rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->beta1, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta1_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta1_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta2, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta2_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta2_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta3, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta3_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta3_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->beta1rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->beta2rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->beta3rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->trK, 0.0);
  deallocate(cctkGH, &cctk_arguments->trK_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->trK_p_p, 0.0);
  deallocate(cctkGH, &cctk_arguments->trKrhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->At11rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->At12rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->At13rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->At22rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->At23rhs, -1.0);
  deallocate(cctkGH, &cctk_arguments->At33rhs, -1.0);
}

static void set_arg(cl_kernel kernel, int arg, cl_mem const* mem)
{
  int ierr = clSetKernelArg(kernel, arg, sizeof(cl_mem), mem);
  assert(!ierr);
}



int exec_ML_BSSN_CL_RHS1(cGH               const* const cctkGH,
                         cctk_arguments_t  const* const cctk_arguments)
{
  
  int ierr;


  int nargs = 0;
  set_arg(kernel1, nargs++, &mem_cctkGH);
  set_arg(kernel1, nargs++, &mem_cctk_parameters);
  set_arg(kernel1, nargs++, &cctk_arguments->x.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->y.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->z.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->r.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At11.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At11_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At11_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At12.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At12_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At12_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At13.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At13_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At13_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At22.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At22_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At22_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At23.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At23_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At23_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At33.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At33_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->At33_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->A.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->A_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->A_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Arhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B1.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B1_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B1_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B2.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B2_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B2_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B3.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B3_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B3_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B1rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B2rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->B3rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt1.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt1_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt1_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt2.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt2_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt2_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt3.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt3_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt3_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt1rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt2rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->Xt3rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->alpha.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->alpha_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->alpha_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->alpharhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->phi.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->phi_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->phi_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->phirhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt11.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt11_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt11_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt12.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt12_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt12_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt13.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt13_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt13_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt22.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt22_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt22_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt23.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt23_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt23_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt33.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt33_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt33_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt11rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt12rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt13rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt22rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt23rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->gt33rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta1.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta1_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta1_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta2.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta2_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta2_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta3.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta3_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta3_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta1rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta2rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->beta3rhs.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->trK.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->trK_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->trK_p_p.mem);
  set_arg(kernel1, nargs++, &cctk_arguments->trKrhs.mem);
  
  size_t const local_work_size[3] =
    { GROUP_SIZE_I, GROUP_SIZE_J, GROUP_SIZE_K };
  size_t const global_work_size[3] =
    {
      divup(cctkGH->lmax[0] - cctkGH->lmin[0],
            VECTOR_SIZE_I * UNROLL_SIZE_I * GROUP_SIZE_I * TILE_SIZE_I) *
      GROUP_SIZE_I,
      divup(cctkGH->lmax[1] - cctkGH->lmin[1],
            VECTOR_SIZE_J * UNROLL_SIZE_J * GROUP_SIZE_J * TILE_SIZE_J) *
      GROUP_SIZE_J,
      divup(cctkGH->lmax[2] - cctkGH->lmin[2],
            VECTOR_SIZE_K * UNROLL_SIZE_K * GROUP_SIZE_K * TILE_SIZE_K) *
      GROUP_SIZE_K,
    };
  {
    static int did_print = 0;
    if (!did_print) {
      did_print = 1;
      printf("Local work group size:  %4d %4d %4d\n",
             (int)local_work_size[0],
             (int)local_work_size[1],
             (int)local_work_size[2]);
      printf("Global work group size: %4d %4d %4d\n",
             (int)global_work_size[0],
             (int)global_work_size[1],
             (int)global_work_size[2]);
    }
  }
  
  ierr = clEnqueueNDRangeKernel(cmd_queue, kernel1, dim,
                                NULL, global_work_size, local_work_size,  
                                0, NULL, NULL);
  assert(!ierr);
  
  ierr = clFinish(cmd_queue);
  assert(!ierr);

  return 0;
}



int exec_ML_BSSN_CL_RHS2(cGH               const* const cctkGH,
                         cctk_arguments_t  const* const cctk_arguments)
{ 
  
  int ierr;

  int nargs = 0;
  set_arg(kernel2, nargs++, &mem_cctkGH);
  set_arg(kernel2, nargs++, &mem_cctk_parameters);
  set_arg(kernel2, nargs++, &cctk_arguments->At11.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At11_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At11_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At12.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At12_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At12_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At13.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At13_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At13_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At22.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At22_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At22_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At23.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At23_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At23_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At33.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At33_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At33_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At11rhs.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At12rhs.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At13rhs.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At22rhs.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At23rhs.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->At33rhs.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt1.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt1_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt1_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt2.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt2_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt2_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt3.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt3_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->Xt3_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->alpha.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->alpha_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->alpha_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->phi.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->phi_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->phi_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt11.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt11_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt11_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt12.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt12_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt12_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt13.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt13_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt13_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt22.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt22_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt22_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt23.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt23_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt23_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt33.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt33_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->gt33_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta1.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta1_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta1_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta2.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta2_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta2_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta3.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta3_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->beta3_p_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->trK.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->trK_p.mem);
  set_arg(kernel2, nargs++, &cctk_arguments->trK_p_p.mem);
  
  size_t const global_work_size[3] =
    { cctkGH->cctk_ash[0], cctkGH->cctk_ash[1], cctkGH->cctk_ash[2] };
  size_t const local_work_size[3] =
    { GROUP_SIZE_I, GROUP_SIZE_J, GROUP_SIZE_K };
  
  ierr = clEnqueueNDRangeKernel(cmd_queue, kernel2, dim,
                                NULL, global_work_size, local_work_size,  
                                0, NULL, NULL);

  assert(!ierr);
  
  ierr = clFinish(cmd_queue);
  assert(!ierr);

  return 0;
}



static void check_var(cGH const* const cctkGH,
                      char const* const name,
                      ptr_t* const ptr,
                      CCTK_REAL const val_int, CCTK_REAL const val_bnd)
{
  int const nsize =
    cctkGH->cctk_ash[0] * cctkGH->cctk_ash[1] * cctkGH->cctk_ash[2];
  int ierr = clEnqueueReadBuffer
    (cmd_queue, ptr->mem, 1, 0, nsize * sizeof(CCTK_REAL), ptr->ptr,
     0, NULL, NULL);
  assert(!ierr);
  for (int k=0; k<cctkGH->cctk_lsh[2]; ++k) {
    for (int j=0; j<cctkGH->cctk_lsh[1]; ++j) {
      for (int i=0; i<cctkGH->cctk_lsh[0]; ++i) {
        int const is_int = (i>=cctkGH->imin[0] && i<cctkGH->imax[0] &&
                            j>=cctkGH->imin[1] && j<cctkGH->imax[1] &&
                            k>=cctkGH->imin[2] && k<cctkGH->imax[2]);
        double const val = is_int ? val_int : val_bnd;
        int const ind3d =
          i + cctkGH->cctk_ash[0] * (j + cctkGH->cctk_ash[1] * k);
        if (! (fabs(ptr->ptr[ind3d] - val) <= 1.0e-15)) {
          printf("%s[%d,%d,%d] is:%.17g should:%.17g\n",
                 name, i,j,k, ptr->ptr[ind3d], val);
        }
        assert(fabs(ptr->ptr[ind3d] - val) <= 1.0e-15);
      }
    }
  }
}



void check(cGH              * const cctkGH,
           cctk_parameters_t* const cctk_parameters,
           cctk_arguments_t * const cctk_arguments)
{
  check_var(cctkGH, "Arhs", &cctk_arguments->Arhs, 0.0, -1.0);
  check_var(cctkGH, "B1rhs", &cctk_arguments->B1rhs, 0.0, -1.0);
  check_var(cctkGH, "B2rhs", &cctk_arguments->B2rhs, 0.0, -1.0);
  check_var(cctkGH, "B3rhs", &cctk_arguments->B3rhs, 0.0, -1.0);
  check_var(cctkGH, "Xt1rhs", &cctk_arguments->Xt1rhs, 0.0, -1.0);
  check_var(cctkGH, "Xt2rhs", &cctk_arguments->Xt2rhs, 0.0, -1.0);
  check_var(cctkGH, "Xt3rhs", &cctk_arguments->Xt3rhs, 0.0, -1.0);
  check_var(cctkGH, "alpharhs", &cctk_arguments->alpharhs, 0.0, -1.0);
  check_var(cctkGH, "phirhs", &cctk_arguments->phirhs, 0.0, -1.0);
  check_var(cctkGH, "gt11rhs", &cctk_arguments->gt11rhs, 0.0, -1.0);
  check_var(cctkGH, "gt12rhs", &cctk_arguments->gt12rhs, 0.0, -1.0);
  check_var(cctkGH, "gt13rhs", &cctk_arguments->gt13rhs, 0.0, -1.0);
  check_var(cctkGH, "gt22rhs", &cctk_arguments->gt22rhs, 0.0, -1.0);
  check_var(cctkGH, "gt23rhs", &cctk_arguments->gt23rhs, 0.0, -1.0);
  check_var(cctkGH, "beta1rhs", &cctk_arguments->beta1rhs, 0.0, -1.0);
  check_var(cctkGH, "beta2rhs", &cctk_arguments->beta2rhs, 0.0, -1.0);
  check_var(cctkGH, "beta3rhs", &cctk_arguments->beta3rhs, 0.0, -1.0);
  check_var(cctkGH, "trKrhs", &cctk_arguments->trKrhs, 0.0, -1.0);
  check_var(cctkGH, "At11rhs", &cctk_arguments->At11rhs, 0.0, -1.0);
  check_var(cctkGH, "At12rhs", &cctk_arguments->At12rhs, 0.0, -1.0);
  check_var(cctkGH, "At13rhs", &cctk_arguments->At13rhs, 0.0, -1.0);
  check_var(cctkGH, "At22rhs", &cctk_arguments->At22rhs, 0.0, -1.0);
  check_var(cctkGH, "At23rhs", &cctk_arguments->At23rhs, 0.0, -1.0);
  check_var(cctkGH, "At33rhs", &cctk_arguments->At33rhs, 0.0, -1.0);
}



#ifndef SRCDIR
#  define SRCDIR "."
#endif

int main(int argc, char** argv)
{
  printf("EinsteinToolkit test\n");

  if (argc > 1)
    if (argv[1][0] == 's')
      use_subdev = 1;

  printf("Reading sources...\n");
  FILE *const source1_file = fopen(SRCDIR "/ML_BSSN_CL_RHS1.cl", "r");
  assert(source1_file != NULL && "ML_BSSN_CL_RHS1.cl not found!");
  fseek(source1_file, 0, SEEK_END);
  size_t const source1_size = ftell(source1_file);
  fseek(source1_file, 0, SEEK_SET);
  char source1[source1_size + 1];
  fread(source1, source1_size, 1, source1_file);
  source1[source1_size] = '\0';
  fclose(source1_file);

  FILE *const source2_file = fopen(SRCDIR "/ML_BSSN_CL_RHS2.cl", "r");
  assert(source2_file != NULL && "ML_BSSN_CL_RHS2.cl not found!");
  fseek(source2_file, 0, SEEK_END);
  size_t const source2_size = ftell(source2_file);
  fseek(source2_file, 0, SEEK_SET);
  char source2[source2_size + 1];
  fread(source2, source2_size, 1, source2_file);
  source2[source2_size] = '\0';
  fclose(source2_file);
  
  printf("Initialise...\n");
  setup(source1, source2);
  cGH cctkGH;
  cctk_parameters_t cctk_parameters;
  cctk_arguments_t cctk_arguments;
  init(&cctkGH, &cctk_parameters, &cctk_arguments);

  printf("RHS1...\n");
  exec_ML_BSSN_CL_RHS1(&cctkGH, &cctk_arguments);
  printf("RHS2...\n");
  exec_ML_BSSN_CL_RHS2(&cctkGH, &cctk_arguments);
  check(&cctkGH, &cctk_parameters, &cctk_arguments);

  printf("Begin timing %d iterations...\n", niters);
  double min_elapsed = HUGE_VAL;
  double avg_elapsed = 0.0;
  for (int n=0; n<niters; ++n) {
    struct timeval tv0;
    gettimeofday(&tv0, NULL);
    exec_ML_BSSN_CL_RHS1(&cctkGH, &cctk_arguments);
    exec_ML_BSSN_CL_RHS2(&cctkGH, &cctk_arguments);
    struct timeval tv1;
    gettimeofday(&tv1, NULL);
    double const elapsed =
      (tv1.tv_sec + 1.0e-6 * tv1.tv_usec) -
      (tv0.tv_sec + 1.0e-6 * tv0.tv_usec);
    min_elapsed = elapsed < min_elapsed ? elapsed : min_elapsed;
    avg_elapsed += elapsed;
  }
  avg_elapsed /= niters;
  printf("End timing\n");
  
  int const npoints =
    cctkGH.cctk_lsh[0] * cctkGH.cctk_lsh[1] * cctkGH.cctk_lsh[2];
  double const time_per_point = min_elapsed / npoints;
  printf("Average elapsed time: %g sec\n", avg_elapsed);
  printf("Minimum elapsed time: %g sec\n", min_elapsed);
  printf("RESULT: Time per grid point update: %g usec\n",
         1.0e+6 * time_per_point);
  double const flop_per_point = 3400.0;
  printf("        This corresponds to %g GFlop/s\n",
         1.0e-9 * flop_per_point / time_per_point);

  printf("\n");
  // VECTOR_SIZE_I=1: 3388 FLop per gpu
  // VECTOR_SIZE_I=2: 3418 Flop per gpu
  printf("Note: This benchmark performs about 3,400  Flop per grid point update.\n");
  printf("      A \"typical\" result is about 1.0 usec.\n");
  printf("      Smaller numbers are better.\n");
  printf("\n");
  
  deinit(&cctkGH, &cctk_parameters, &cctk_arguments);
  cleanup();
  
  printf ("Done.\n");
  return 0;
}
