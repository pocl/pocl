#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#include <android/log.h>
#include "vectorAdd.h"

#define LOCAL_SIZE  64


#define CHECK_AND_RETURN(ret, msg)                                          \
    if(ret != CL_SUCCESS) {                                                 \
        __android_log_print(ANDROID_LOG_ERROR, "opencl vector add",         \
				"ERROR: %s at line %d in %s returned with %d\n",            \
					msg, __LINE__, __FILE__, ret);                          \
        return ret;                                                         \
    }


static const char *vector_add_str="											\
		__kernel void vec_add(int N, __global float *A,						\
								__global float *B, __global float *C)		\
		{																	\
			int id = get_global_id(0);										\
																			\
			if(id < N) {													\
				C[id] = A[id] + B[id];										\
			}																\
		}																	\
		";

static cl_context clContext = NULL;
static cl_command_queue clCommandQueue = NULL;
static cl_program clProgram = NULL;
static cl_kernel clKernel = NULL;

jint Java_org_pocl_sample1_MainActivity_initCL(JNIEnv *je, jobject jo)
{
	cl_platform_id clPlatform;
	cl_device_id clDevice;
    cl_int	status;

    status = clGetPlatformIDs(1, &clPlatform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

	status = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_DEFAULT, 1, &clDevice, NULL);
    CHECK_AND_RETURN(status, "getting device id failed");

    cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) clPlatform,
                                      0 };

    clContext = clCreateContext(cps, 1, &clDevice, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

	clCommandQueue = clCreateCommandQueue(clContext, clDevice, 0, &status);
    CHECK_AND_RETURN(status, "creating command queue failed");

    size_t strSize = strlen(vector_add_str);
    clProgram = clCreateProgramWithSource(clContext, 1, &vector_add_str, &strSize, &status);
    CHECK_AND_RETURN(status, "creating program failed");

	status = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "build program failed");

	clKernel = clCreateKernel(clProgram, "vec_add", &status);
    CHECK_AND_RETURN(status, "creating kernel failed");

    return 0;
}


jint Java_org_pocl_sample1_MainActivity_destroyCL(JNIEnv *je, jobject jo)
{
	if(clKernel)		clReleaseKernel(clKernel);
	if(clProgram)		clReleaseProgram(clProgram);
	if(clCommandQueue) 	clReleaseCommandQueue(clCommandQueue);
	if(clContext)	    clReleaseContext(clContext);

    return 0;
}


jint Java_org_pocl_sample1_MainActivity_vectorAddCL(JNIEnv *je , jobject jo,
						jint N, jfloatArray _A, jfloatArray _B, jfloatArray _C)
{
    cl_int	status;
    int byteSize = N * sizeof(float);

    // Get pointers to array from jni wrapped floatArray
    jfloat* A = je->GetFloatArrayElements(_A, 0);
    jfloat* B = je->GetFloatArrayElements(_B, 0);
    jfloat* C = je->GetFloatArrayElements(_C, 0);

    cl_mem A_obj = clCreateBuffer(clContext, (CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR),
                                                    byteSize, A, &status);
    CHECK_AND_RETURN(status, "create buffer A failed");

    cl_mem B_obj = clCreateBuffer(clContext, (CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR),
                                                    byteSize, B, &status);
    CHECK_AND_RETURN(status, "create buffer B failed");

    cl_mem C_obj = clCreateBuffer(clContext, (CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR),
                                                    byteSize, C, &status);
    CHECK_AND_RETURN(status, "create buffer C failed");

    status = clSetKernelArg(clKernel, 0, sizeof(cl_int), (void *)&N);
    status |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&A_obj);
    status |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&B_obj);
    status |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&C_obj);
    CHECK_AND_RETURN(status, "clSetKernelArg failed");

    size_t localSize = LOCAL_SIZE;
    size_t wgs = (N + localSize - 1) / localSize;
    size_t globalSize = wgs * localSize;

    status = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL,
                            &globalSize, &localSize, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "clEnqueueNDRange failed");

    status = clFinish(clCommandQueue);
    CHECK_AND_RETURN(status, "clFinish failed");

    je->ReleaseFloatArrayElements(_A, A, 0);
    je->ReleaseFloatArrayElements(_B, B, 0);
    je->ReleaseFloatArrayElements(_C, C, 0);

    return 0;
}

void Java_org_pocl_sample1_MainActivity_setenv(JNIEnv *jniEnv,
						jobject _jObj, jstring key, jstring value)
{
	setenv((char*) jniEnv->GetStringUTFChars(key, 0),
			(char*) jniEnv->GetStringUTFChars(value, 0), 1);
}



