#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2024 PoCL Developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

# do not link test_dlopen with -lOpenCL
if (UNIX AND HAVE_DLFCN_H)
  add_executable("test_dlopen" "test_dlopen.c")
  if(SANITIZER_OPTIONS)
    target_link_libraries("test_dlopen" ${SANITIZER_LIBS})
  endif()
  target_link_libraries("test_dlopen" ${DL_LIB})
endif ()

include_directories(${CMAKE_SOURCE_DIR})

set(C_PROGRAMS_TO_BUILD test_clFinish test_clGetDeviceInfo test_clGetEventInfo
  test_clCreateProgramWithBinary test_clGetSupportedImageFormats
  test_clSetEventCallback test_clEnqueueNativeKernel test_clBuildProgram
  test_clCreateKernelsInProgram test_clCreateKernel test_clGetKernelArgInfo
  test_version test_kernel_cache_includes test_event_cycle test_link_error
  test_read-copy-write-buffer test_buffer-image-copy test_clCreateSubDevices test_event_free
  test_event_double_wait test_buffer_migration test_buffer_ping_pong
  test_enqueue_kernel_from_binary test_user_event test_fill-buffer
  test_clSetMemObjectDestructorCallback test_dbk_jpeg
  test_cl_pocl_content_size test_cl_pocl_content_size_migration
  test_deviceside_enqueue test_command_buffer test_command_buffer_images
  test_command_buffer_multi_device test_queue_creation_with_hints)

if(HAVE_ONNXRT)
  list(APPEND C_PROGRAMS_TO_BUILD test_dbk_onnx_inference)
endif()

set(CXX_PROGRAMS_TO_BUILD test_device_address test_svm test_large_buf
  test_subbuffers test_compile_n_link test_dbk_matmul)

add_compile_options(${OPENCL_CFLAGS} -I${CMAKE_SOURCE_DIR}/include)

foreach(PROG ${C_PROGRAMS_TO_BUILD})
  if(MSVC)
    set_source_files_properties( "${PROG}.c" PROPERTIES LANGUAGE CXX )
  endif(MSVC)
  add_executable("${PROG}" "${PROG}.c")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()

foreach(PROG ${CXX_PROGRAMS_TO_BUILD})
  set_source_files_properties( "${PROG}.cpp" PROPERTIES LANGUAGE CXX )
  add_executable("${PROG}" "${PROG}.cpp")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()

#######################################################################


add_test_pocl(NAME "runtime/clGetDeviceInfo" COMMAND "test_clGetDeviceInfo" WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/clEnqueueNativeKernel" COMMAND "test_clEnqueueNativeKernel")

add_test_pocl(NAME "runtime/clGetEventInfo" COMMAND "test_clGetEventInfo" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/clCreateProgramWithBinary" COMMAND "test_clCreateProgramWithBinary" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/clBuildProgram"
              WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
              COMMAND "test_clBuildProgram"
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_kernel_cache_includes"
              WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
              COMMAND "test_kernel_cache_includes"
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/clFinish" COMMAND "test_clFinish" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_event_cycle" COMMAND "test_event_cycle" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_link_error" COMMAND "test_link_error" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_read-copy-write-buffer" COMMAND "test_read-copy-write-buffer" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_fill-buffer" COMMAND "test_fill-buffer" WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/test_buffer-image-copy" COMMAND "test_buffer-image-copy")

add_test_pocl(NAME "runtime/clCreateKernel" COMMAND "test_clCreateKernel" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/clGetKernelArgInfo" COMMAND "test_clGetKernelArgInfo" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/clSetEventCallback"
              COMMAND "test_clSetEventCallback"
              EXPECTED_OUTPUT "test_clSetEventCallback_expout.txt"
              WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/clGetSupportedImageFormats" COMMAND "test_clGetSupportedImageFormats")

add_test_pocl(NAME "runtime/clCreateKernelsInProgram" COMMAND "test_clCreateKernelsInProgram" WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/clCreateSubDevices" COMMAND  "test_clCreateSubDevices")

add_test_pocl(NAME "runtime/test_event_free" COMMAND  "test_event_free" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_event_double_wait" COMMAND  "test_event_double_wait" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_enqueue_kernel_from_binary" COMMAND "test_enqueue_kernel_from_binary" WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "runtime/test_user_event" COMMAND  "test_user_event" WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/test_buffer_migration" COMMAND "test_buffer_migration")

add_test(NAME "runtime/test_buffer_ping_pong" COMMAND "test_buffer_ping_pong")

add_test_pocl(NAME "runtime/clSetMemObjectDestructorCallback" COMMAND  "test_clSetMemObjectDestructorCallback" WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/test_cl_pocl_content_size" COMMAND "test_cl_pocl_content_size")

add_test_pocl(NAME "runtime/test_deviceside_enqueue" COMMAND "test_deviceside_enqueue" WORKITEM_HANDLER "loopvec")

add_test(NAME "runtime/test_command_buffer" COMMAND "test_command_buffer")

add_test(NAME "runtime/test_command_buffer_images" COMMAND "test_command_buffer_images")

add_test(NAME "runtime/test_command_buffer_multi_device" COMMAND "test_command_buffer_multi_device")

add_test(NAME "runtime/test_device_address" COMMAND "test_device_address")

add_test(NAME "runtime/test_svm" COMMAND "test_svm")

add_test(NAME "runtime/test_subbuffers" COMMAND "test_subbuffers")

add_test(NAME "runtime/test_large_buf" COMMAND "test_large_buf")

add_test(NAME "runtime/test_compile_n_link" COMMAND "test_compile_n_link")

if(HAVE_LIBJPEG_TURBO)
  add_test(NAME "runtime/test_dbk_jpeg"
    COMMAND test_dbk_jpeg 640 480
    "${CMAKE_CURRENT_SOURCE_DIR}/test_data/tram.rgb")
  set_tests_properties("runtime/test_dbk_jpeg"
    PROPERTIES
    COST 2.0
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;runtime")
endif()

set_property(TEST
  "runtime/test_large_buf"
  APPEND PROPERTY LABELS "mingw_fail")

add_test(NAME "runtime/test_queue_creation_with_hints" COMMAND "test_queue_creation_with_hints")

add_test(NAME "runtime/test_dbk_matmul" COMMAND test_dbk_matmul)

if(HAVE_ONNXRT)
  add_test(NAME "runtime/test_dbk_onnx_inference" COMMAND "test_dbk_onnx_inference")
endif()

set_tests_properties( "runtime/clGetDeviceInfo" "runtime/clEnqueueNativeKernel"
  "runtime/clGetEventInfo" "runtime/clCreateProgramWithBinary"
  "runtime/clBuildProgram" "runtime/clFinish" "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats" "runtime/clCreateKernelsInProgram"
  "runtime/clCreateKernel" "runtime/clGetKernelArgInfo"
  "runtime/test_kernel_cache_includes" "runtime/test_event_cycle"
  "runtime/test_read-copy-write-buffer" "runtime/test_buffer-image-copy"
  "runtime/test_fill-buffer"
  "runtime/test_event_free" "runtime/test_event_double_wait" "runtime/clCreateSubDevices"
  "runtime/test_enqueue_kernel_from_binary" "runtime/test_user_event"
  "runtime/test_buffer_migration"
  "runtime/test_buffer_ping_pong"
  "runtime/clSetMemObjectDestructorCallback" "runtime/test_link_error"
  "runtime/test_cl_pocl_content_size" "runtime/test_deviceside_enqueue"
  "runtime/test_command_buffer" "runtime/test_command_buffer_images"
  "runtime/test_command_buffer_multi_device"
  "runtime/test_device_address" "runtime/test_svm"
  "runtime/test_device_address" "runtime/test_svm"
  "runtime/test_compile_n_link"
  "runtime/test_subbuffers"
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;runtime")

set_tests_properties(
  "runtime/clCreateSubDevices"
  "runtime/test_buffer_migration"
  "runtime/test_buffer_ping_pong"
  "runtime/test_cl_pocl_content_size"
  "runtime/test_buffer-image-copy"
  "runtime/clGetSupportedImageFormats"
  "runtime/clEnqueueNativeKernel"
  "runtime/test_command_buffer"
  "runtime/test_command_buffer_images"
  "runtime/test_command_buffer_multi_device"
  "runtime/test_device_address"
  "runtime/test_svm"
  "runtime/test_large_buf"
  "runtime/test_dbk_matmul"
  PROPERTIES SKIP_RETURN_CODE 77)

if(ENABLE_REMOTE_CLIENT AND ENABLE_REMOTE_SERVER AND ENABLE_HOST_CPU_DEVICES)
  add_test(NAME "remote/clGetDeviceInfo"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clGetDeviceInfo")
  add_test(NAME "remote/clEnqueueNativeKernel"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clEnqueueNativeKernel")
  add_test(NAME "remote/clGetEventInfo"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clGetEventInfo")
  add_test(NAME "remote/clCreateProgramWithBinary"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clCreateProgramWithBinary")
  add_test(NAME "remote/test_event_cycle"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_event_cycle")
  add_test(NAME "remote/test_link_error"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_link_error")
  add_test(NAME "remote/test_read-copy-write-buffer"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_read-copy-write-buffer")
  add_test(NAME "remote/test_fill-buffer"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_fill-buffer")
  add_test(NAME "remote/test_buffer-image-copy"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_buffer-image-copy")
  add_test(NAME "remote/clCreateKernel"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clCreateKernel")
  add_test(NAME "remote/clGetKernelArgInfo"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clGetKernelArgInfo")
  add_test(NAME "remote/clGetSupportedImageFormats"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clGetSupportedImageFormats")
  add_test(NAME "remote/clCreateKernelsInProgram"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clCreateKernelsInProgram")
  add_test(NAME "remote/clCreateSubDevices"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clCreateSubDevices")
  add_test(NAME "remote/test_event_free"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_event_free")
  add_test(NAME "remote/test_event_double_wait"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_event_double_wait")
  add_test(NAME "remote/test_enqueue_kernel_from_binary"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_enqueue_kernel_from_binary")
  add_test(NAME "remote/test_user_event"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_user_event")
  add_test(NAME "remote/test_buffer_migration"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_multi.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_buffer_migration")
  add_test(NAME "remote/test_buffer_ping_pong"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_multi.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_buffer_ping_pong")
  add_test(NAME "remote/clSetMemObjectDestructorCallback"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_clSetMemObjectDestructorCallback")
  add_test(NAME "remote/test_cl_pocl_content_size"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_cl_pocl_content_size")
  add_test(NAME "remote/test_cl_pocl_content_size_migration_host_remote"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_with_local.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_cl_pocl_content_size_migration")
  add_test(NAME "remote/test_cl_pocl_content_size_migration_remote_remote"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_multi.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_cl_pocl_content_size_migration")
  add_test(NAME "remote/test_deviceside_enqueue"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_deviceside_enqueue")
  add_test(NAME "remote/test_command_buffer"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_command_buffer")
  add_test(NAME "remote/test_command_buffer_images"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_command_buffer_images")
  add_test(NAME "remote/test_command_buffer_multi_device"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_multi.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_command_buffer_multi_device")
  add_test(NAME "remote/test_device_address"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_device_address")
  add_test(NAME "remote/test_svm"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_multi.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_svm")
  add_test(NAME "remote/test_queue_creation_with_hints"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "tests/runtime/test_queue_creation_with_hints")

  set_tests_properties(
    "remote/clCreateSubDevices"
    "remote/test_buffer_migration"
    "remote/test_buffer_ping_pong"
    "remote/test_cl_pocl_content_size"
    "remote/test_cl_pocl_content_size_migration_host_remote"
    "remote/test_cl_pocl_content_size_migration_remote_remote"
    "remote/test_buffer-image-copy"
    "remote/clGetSupportedImageFormats"
    "remote/clEnqueueNativeKernel"
    "remote/test_command_buffer"
    "remote/test_command_buffer_images"
    "remote/test_command_buffer_multi_device"
    "remote/test_device_address"
    "remote/test_svm"
    "remote/test_queue_creation_with_hints"
    PROPERTIES SKIP_RETURN_CODE 77)

  set_property(TEST "remote/test_svm"
    APPEND PROPERTY ENVIRONMENT "POCLD_COARSE_GRAIN_SVM=1 POCLD_COARSE_GRAIN_SVM_MAX_SIZE=10")

  set_tests_properties(
    "remote/clGetDeviceInfo" "remote/clEnqueueNativeKernel"
    "remote/clGetEventInfo" "remote/clCreateProgramWithBinary"
    "remote/clGetSupportedImageFormats" "remote/clCreateKernel"
    "remote/clGetKernelArgInfo" "remote/test_event_cycle"
    "remote/test_read-copy-write-buffer" "remote/test_buffer-image-copy"
    "remote/test_fill-buffer" "remote/test_event_free"
    "remote/test_event_double_wait" "remote/clCreateSubDevices"
    "remote/test_enqueue_kernel_from_binary" "remote/test_user_event"
    "remote/test_buffer_migration" "remote/test_buffer_ping_pong"
    "remote/clSetMemObjectDestructorCallback" "remote/test_link_error"
    "remote/test_cl_pocl_content_size"
    "remote/test_cl_pocl_content_size_migration_host_remote"
    "remote/test_cl_pocl_content_size_migration_remote_remote"
    "remote/test_deviceside_enqueue"
    "remote/test_command_buffer" "remote/test_command_buffer_images"
    "remote/test_command_buffer_multi_device"
    "remote/test_device_address" "remote/test_svm"
    "remote/test_queue_creation_with_hints"
    PROPERTIES
      PASS_REGULAR_EXPRESSION "OK"
      COST 2.0
      PROCESSORS 1
      DEPENDS "pocl_version_check"
      LABELS "remote"
      RESOURCE_LOCK "pocld_ports")

  if(ENABLE_RDMA)
    set_tests_properties(
      "remote/test_buffer_migration"
      "remote/test_buffer_ping_pong"
      "remote/test_cl_pocl_content_size_migration_remote_remote"
      PROPERTIES
        LABELS "remote_rdma")
  endif()
endif()

if(NOT ENABLE_ANYSAN)
  set_tests_properties("runtime/clCreateKernelsInProgram"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "Hello\nWorld")

  set_tests_properties("runtime/clFinish"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "ABABC;A\nB\nA\nB\nC\n")

  set_tests_properties("runtime/test_kernel_cache_includes"
  PROPERTIES PASS_REGULAR_EXPRESSION
  "function 1.*first include.*function 2.*second include")
endif()

if(ENABLE_ASAN)
  set_tests_properties("runtime/clCreateSubDevices"
    PROPERTIES DISABLED 1)
endif()

if(NOT HAVE_WORKING_IMAGES_WITH_SPIR)
  set_tests_properties("runtime/clGetKernelArgInfo"
    PROPERTIES DISABLED 1)
endif()

# Label tests that work with CUDA backend
set_property(TEST
  "runtime/clGetDeviceInfo"
  "runtime/clGetEventInfo"
  "runtime/clCreateProgramWithBinary"
  "runtime/test_kernel_cache_includes"
  "runtime/clFinish"
  "runtime/test_read-copy-write-buffer"
  "runtime/test_fill-buffer"
  "runtime/test_buffer-image-copy"
  "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats"
  "runtime/clCreateKernelsInProgram"
  "runtime/test_event_cycle"
  "runtime/test_event_free"
  "runtime/test_event_double_wait"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  "runtime/test_deviceside_enqueue"
  "runtime/test_device_address"
  "runtime/test_compile_n_link"
  "runtime/test_svm"
  "runtime/test_subbuffers"
  "runtime/test_dbk_matmul"
  "runtime/test_queue_creation_with_hints"
  APPEND PROPERTY LABELS "cuda")

# Label tests that work with TCE backend
set_property(TEST
  "runtime/test_fill-buffer"
  "runtime/test_dbk_matmul"
  "runtime/test_queue_creation_with_hints"
  APPEND PROPERTY LABELS "tce")


set_property(TEST
  "runtime/clGetDeviceInfo"
  "runtime/clGetEventInfo"
  "runtime/clCreateProgramWithBinary"
  "runtime/clBuildProgram"
  "runtime/test_kernel_cache_includes"
  "runtime/test_event_cycle"
  "runtime/test_link_error"
  "runtime/test_buffer-image-copy"
  "runtime/clCreateKernel"
  "runtime/clGetKernelArgInfo"
  "runtime/clGetSupportedImageFormats"
  "runtime/clSetEventCallback"
  "runtime/clCreateKernelsInProgram"
  "runtime/clFinish"
  "runtime/test_enqueue_kernel_from_binary"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  "runtime/test_deviceside_enqueue"
  "runtime/test_dbk_matmul"
  "runtime/test_queue_creation_with_hints"
  APPEND PROPERTY LABELS "hsa-native")

set_property(TEST
  "runtime/clGetEventInfo"
  "runtime/clFinish"
  "runtime/test_read-copy-write-buffer"
  "runtime/test_fill-buffer"
  "runtime/test_buffer-image-copy"
  "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats"
  "runtime/test_event_cycle"
  "runtime/test_event_free"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  "runtime/test_dbk_matmul"
  "runtime/test_queue_creation_with_hints"
  APPEND PROPERTY LABELS "proxy")

# Label tests that work with Vulkan
set_property(TEST
  "runtime/clGetEventInfo"
  "runtime/test_event_cycle"
  "runtime/test_event_free"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  "runtime/test_dbk_matmul"
  "runtime/test_queue_creation_with_hints"
  APPEND PROPERTY LABELS "vulkan")

if (ENABLE_TCE)
    # Label tests that work with Almaif (with compiler) backend
    set_property(TEST
        "runtime/test_fill-buffer"
        "runtime/test_dbk_matmul"
        "runtime/test_queue_creation_with_hints"
        APPEND PROPERTY LABELS "almaif")
endif()

# all except "runtime/test_command_buffer_images" "runtime/clGetKernelArgInfo"
set_property(TEST
  "runtime/clGetDeviceInfo" "runtime/clEnqueueNativeKernel"
  "runtime/clGetEventInfo" "runtime/clCreateProgramWithBinary"
  "runtime/clBuildProgram" "runtime/clFinish" "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats" "runtime/clCreateKernelsInProgram"
  "runtime/clCreateKernel"
  "runtime/test_kernel_cache_includes" "runtime/test_event_cycle"
  "runtime/test_read-copy-write-buffer" "runtime/test_buffer-image-copy"
  "runtime/test_fill-buffer"
  "runtime/test_event_free" "runtime/test_event_double_wait" "runtime/clCreateSubDevices"
  "runtime/test_enqueue_kernel_from_binary" "runtime/test_user_event"
  "runtime/test_buffer_migration"
  "runtime/test_buffer_ping_pong"
  "runtime/clSetMemObjectDestructorCallback" "runtime/test_link_error"
  "runtime/test_cl_pocl_content_size" "runtime/test_deviceside_enqueue"
  "runtime/test_command_buffer"
  "runtime/test_device_address"
  "runtime/test_svm"
  "runtime/test_subbuffers"
  "runtime/test_dbk_matmul"
  "runtime/test_queue_creation_with_hints"
  APPEND PROPERTY LABELS "level0")

if(ENABLE_LEVEL0 AND ENABLE_HOST_CPU_DEVICES)
  add_test(NAME "runtime/test_subbuffers_multidevice" COMMAND "test_subbuffers")
  set_property(TEST "runtime/test_subbuffers_multidevice"
    APPEND PROPERTY ENVIRONMENT "POCL_DEVICES=level0 cpu")

endif()
