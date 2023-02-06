/**
 * \brief OpenCL runtime library: poclu - useful utility functions for OpenCL programs.

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

   \file
*/

#ifndef POCLU_H
#define POCLU_H

#include "pocl_opencl.h"

#ifdef _WIN32
#define POCLU_CALL __cdecl
#define POCLU_API __declspec(dllexport)
#else
#define POCLU_CALL
#define POCLU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
* \brief Byte swap functions for endianness swapping between the host
* (current CPU) and a target device.
*
* Queries the target device using the OpenCL API for the endianness
* and swaps if it differs from the host's.
*
* \param [in] device the device id of the target
* \param [in] original the input to potentially swapped
* \returns the input in the correct endianness.
*
*/
POCLU_API cl_int POCLU_CALL
poclu_bswap_cl_int(cl_device_id device, cl_int original);

/**
 * \brief Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 *
 * \param [in] device the device id of the target
 * \param [in] original the input to potentially swapped
 * \returns the input in the correct endianness.
 *
 */
POCLU_API cl_half POCLU_CALL
poclu_bswap_cl_half(cl_device_id device, cl_half original);

/**
* \brief Byte swap functions for endianness swapping between the host
* (current CPU) and a target device.
*
* Queries the target device using the OpenCL API for the endianness
* and swaps if it differs from the host's.
*
* \param [in] device the device id of the target
* \param [in] original the input to potentially swapped
* \returns the input in the correct endianness.
*
*/
POCLU_API cl_float POCLU_CALL
poclu_bswap_cl_float(cl_device_id device, cl_float original);

/**
* \brief Byte swap functions for endianness swapping between the host
* (current CPU) and a target device.
*
* Queries the target device using the OpenCL API for the endianness
* and swaps if it differs from the host's.
*
* \param [in] device the device id of the target
* \param [in] original the input to potentially swapped
* \returns the input in the correct endianness.
*
*/
POCLU_API cl_float2 POCLU_CALL
poclu_bswap_cl_float2(cl_device_id device, cl_float2 original);


/**
 * \brief Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 *
 * \param [in] device the device id of the target
 * \param [in] array array pointer to elements to be potentially swapped
 * \param [in] num_elements number of elements in the array
 * \returns the input in the correct endianness
 *
 */
POCLU_API void POCLU_CALL
poclu_bswap_cl_int_array(cl_device_id device, cl_int* array, size_t num_elements);

/**
 * \brief Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 *
 * \param [in] device the device id of the target
 * \param [in] array array pointer to elements to be potentially swapped
 * \param [in] num_elements number of elements in the array
 * \returns the input in the correct endianness
 *
 */
POCLU_API void POCLU_CALL
poclu_bswap_cl_half_array(cl_device_id device, cl_half* array, size_t num_elements);

/**
 * \brief Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 *
 * \param [in] device the device id of the target
 * \param [in] array array pointer to elements to be potentially swapped
 * \param [in] num_elements number of elements in the array
 * \returns the input in the correct endianness
 *
 */
POCLU_API void POCLU_CALL
poclu_bswap_cl_float_array(cl_device_id device, cl_float* array, size_t num_elements);

/**
 * \brief Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 *
 * \param [in] device the device id of the target
 * \param [in] array array pointer to elements to be potentially swapped
 * \param [in] num_elements number of elements in the array
 * \returns the input in the correct endianness
 *
 */
POCLU_API void POCLU_CALL
poclu_bswap_cl_float2_array(cl_device_id device, cl_float2* array, size_t num_elements);

/**
 * Misc. helper functions for streamlining OpenCL API usage.
 */

/**
 * \brief Create a context in the first platform found.
 * \returns a cl_context.
 */
POCLU_API cl_context POCLU_CALL
poclu_create_any_context();

/**
 * \brief Set up a context, device and queue for platform 0, device 0.
 *
 * similar to pocl_get_any_device, but returns the platform as well.
 * \warning All input parameters must be allocated by caller!
 * @param context [out]
 * @param device [out]
 * @param queue [out]
 * @param platform [out]
 * \return CL_SUCCESS on success, or a descriptive OpenCL error code upon failure.
 */
POCLU_API cl_int POCLU_CALL poclu_get_any_device2 (cl_context *context,
                                                   cl_device_id *device,
                                                   cl_command_queue *queue,
                                                   cl_platform_id *platform);
/**
 * \brief set up a context, device and queue for platform 0, device 0.
 *
 * similar to pocl_get_any_device2, but doesn't return the platform.
 * @param context [out]
 * @param device [out]
 * @param queue [out]
 * \return CL_SUCCESS on success, or a descriptive OpenCL error code upon failure.
 * \warning All input parameters must be allocated by caller!
 */
POCLU_API cl_int POCLU_CALL poclu_get_any_device (cl_context *context,
                                                  cl_device_id *device,
                                                  cl_command_queue *queue);

/**
 * \brief set up all available devices and queues for a single context on
 * the first platform.
 * @param platform [out] pointer to the first platform found.
 * @param context [out] pointer to the shared context of all devices.
 * @param num_devices [out] the number of devices found.
 * @param devices [out] pointer to array of device pointers.
 * @param queues [out] pointer to array of queue pointers.
 * @return CL_SUCCESS on success, or a descriptive OpenCL error code upon failure.
 */
POCLU_API cl_int POCLU_CALL poclu_get_multiple_devices (
    cl_platform_id *platform, cl_context *context,
    cl_char include_custom_dev, cl_uint *num_devices,
    cl_device_id **devices, cl_command_queue **queues);

/**
 * \brief convert a float to a cl_half (uint16_t).
 *
 * The idea behind these float to half functions is from:
 * https://gamedev.stackexchange.com/a/17410
 * @param value [in] float to be converted.
 * @return a converted cl_half.
 */
POCLU_API cl_half POCLU_CALL
poclu_float_to_cl_half(float value);

/**
 * \brief convert a cl_half to a float.
 * @param value [in] cl_half to be converted.
 * @return a converted float.
 */
POCLU_API float POCLU_CALL
poclu_cl_half_to_float(cl_half value);

/**
 * \brief read the contents of a file.
 *
 * filename can absolute or relative, according to the fopen
 * implementation.
 * @param filename [in] string to the file.
 * @return a malloc'd buffer pointer or NULL on errors.
 */
POCLU_API char *POCLU_CALL poclu_read_file (const char *filename);

/**
 * \brief check if opencl device(s) support OpenCL 3.0
 *
 * @param devices [in] array of SAME devices for program to run on
 * @param num_devices [in] number of devices in array of devices.
 * @return 1 if all devices support opencl 30, 0 otherwise.
 */
POCLU_API int POCLU_CALL poclu_supports_opencl_30 (cl_device_id *devices,
                                                   unsigned num_devices);

/**
 * \brief read the contents of a file.
 *
 * filename can absolute or relative, according to the fopen
 * implementation.
 * @param filename [in] string to the file.
 * @param len [out] length of the binary file.
 * @return a malloc'd buffer pointer or NULL on errors.
 */
POCLU_API char *POCLU_CALL poclu_read_binfile (const char *filename,
                                               size_t *len);
/**
 * \brief write content to a file.
 *
 * filename can absolute or relative, according to the fopen
 * implementation on system.
 * @param filename [in] string to the file.
 * @param content [in] string to be written.
 * @param size [in] size of the content.
 * @return -1 if there is any error otherwise 0.
 */
POCLU_API int POCLU_CALL poclu_write_file (const char *filename, char *content,
                                           size_t size);
/**
 * \brief wrapper for poclu_load_program_multidev, see it for details.
 */
int poclu_load_program (cl_context context, cl_device_id device,
                        const char *basename, int spir, int spirv, int poclbin,
                        const char *explicit_binary,
                        const char *extra_build_opts, cl_program *p);
/**
 * \brief create a program from different sources
 *
 * this function can create a program from a wide variety of sources,
 * including: spir, spirv, cl or poclbin.
 * It will look for a file with a name matching basename + source type in
 * the following order of directories: current, build and source.
 * If the explicit_binary option is given, it will use that instead.
 * NOTE that the file extension needs to be included in the explicit_binary
 * option.
 * @param context [in] context where the program should run
 * @param devices [in] array of SAME devices for program to run on
 * @param num_devices [in] number of devices in array of devices.
 * @param basename [in] name of the program to be run, used to find sources.
 * @param spir [in] boolean to indicate whether to use spir sources.
 * @param spirv [in] boolean to indicate whether to use spirv sources.
 * @param poclbin [in] boolean to indicate whether to use a pocl binary source.
 * @param explicit_binary [in] optional explicit path to sources.
 * @param extra_build_opts [in] string of extra options to use when compiling sources.
 * @param p [out] resulting program from source.
 * @return CL_SUCCESS or else a related CL error.
 *
 * \warning all devices need to be of the same type or else an error will be thrown.
 */
int poclu_load_program_multidev (cl_context context, cl_device_id *devices,
                                 cl_uint num_devices, const char *basename,
                                 int spir, int spirv, int poclbin,
                                 const char *explicit_binary,
                                 const char *extra_build_opts, cl_program *p);

/**
 * \brief print program build log of each device to stderr.
 * @param program [in] program of which to print build logs.
 * @return CL_SUCCESS or else a CL related error.
 */
POCLU_API cl_int POCLU_CALL poclu_show_program_build_log (cl_program program);

/**
 * \brief check a return value of CL function or print the error to stderr.
 * @param cl_err [in] value to be checked.
 * @param line  [in] line of function in question.
 * @param func_name [in] name of the function in question.
 * @return 0 if cl_err is CL_SUCCESS otherwise 1.
 */
POCLU_API int POCLU_CALL check_cl_error (cl_int cl_err, int line,
                                         const char *func_name);

/**
 * \brief private macro to check error code and return EXIT_FAILURE on error
 */
#define _POCLU_CHECK_CL_ERROR_INNER(cond, func, line)                         \
  do                                                                          \
    {                                                                         \
      if (check_cl_error (cond, line, func))                                  \
        return (EXIT_FAILURE);                                                \
    }                                                                         \
  while (0)

/**
 * \brief check a CL return value and if it is an error: print it and exit function on failure
 * @param cond [in] cl_int value to check.
 */
#define CHECK_CL_ERROR(cond) _POCLU_CHECK_CL_ERROR_INNER(cond, __PRETTY_FUNCTION__, __LINE__)

/**
 * \brief check a CL return value and if it is an error: print the message and exit function on failure.
 * @param message [in] string to print when error is unknown.
 * \warning macro assumes CL return value is called "err"
 */
#define CHECK_OPENCL_ERROR_IN(message) _POCLU_CHECK_CL_ERROR_INNER(err, message, __LINE__)

/**
 * \brief check an expression and if false: print to stderr and exit of failure
 * @param EXP [in]
 */
#define TEST_ASSERT(EXP)                                                \
do {                                                                    \
  if (!(EXP)) {                                                         \
    fprintf(stderr, "Assertion: \n" #EXP "\nfailed on %s:%i\n",         \
        __FILE__, __LINE__);                                            \
    return EXIT_FAILURE;                                                \
  }                                                                     \
} while (0)

/**
 * \brief check CL return value and print to stderr and jump to ERROR statement
 * on failure
 * \warning macro assumes "ERROR" statement exists
 */
#define CHECK_CL_ERROR2(err)                                                  \
  if (check_cl_error (err, __LINE__, __PRETTY_FUNCTION__))                    \
  goto ERROR

#ifdef __cplusplus
}
#endif

#endif
