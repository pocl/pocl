#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2015 pocl developers
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

set(TS_NAME "piglit")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

find_package(GLUT QUIET)
find_package(PNG QUIET)

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
elseif(NOT GLUT_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires glut library")
elseif(NOT PNG_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires PNG library")
elseif(NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd")

else()
message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

#-- Found PythonInterp: /usr/bin/python2.7 (found suitable version "2.7.11", minimum required is "2.7")
#-- Found PythonNumpy: success (found suitable version "1.10.2", minimum required is "1.6.2")
#-- Found PythonMako: success (found suitable version "1.0.3", minimum required is "0.8.0")
#-- Found PythonSix: success (found suitable version "1.10.0", minimum required is "1.4.0")

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "git://anongit.freedesktop.org/piglit"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
    -DPIGLIT_BUILD_CL_TESTS=ON
    -DPIGLIT_BUILD_DMA_BUF_TESTS:BOOL=OFF
    -DPIGLIT_BUILD_GL_TESTS=OFF
    -DPIGLIT_BUILD_GLES1_TESTS=OFF
    -DPIGLIT_BUILD_GLES2_TESTS=OFF
    -DPIGLIT_BUILD_GLES3_TESTS=OFF
    -DPIGLIT_USE_WAFFLE=OFF
    -DPIGLIT_BUILD_GLX_TESTS=OFF
    "-DOPENCL_INCLUDE_PATH=${CMAKE_SOURCE_DIR}/include/"
    "-DOPENCL_opencl_LIBRARY:LIST=OpenCL"
  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME piglit_cl_api_build_program
         COMMAND "${TS_BUILDDIR}/bin/cl-api-build-program")
add_test(NAME piglit_cl_api_compile_program
         COMMAND "${TS_BUILDDIR}/bin/cl-api-compile-program")
add_test(NAME piglit_cl_api_create_buffer
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-buffer")
add_test(NAME piglit_cl_api_create_command_queue
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-command-queue")
add_test(NAME piglit_cl_api_create_context
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-context")
add_test(NAME piglit_cl_api_create_context_from_type
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-context-from-type")
add_test(NAME piglit_cl_api_create_image
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-image")
add_test(NAME piglit_cl_api_create_kernel
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-kernel")
add_test(NAME piglit_cl_api_create_kernels_in_program
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-kernels-in-program")
add_test(NAME piglit_cl_api_create_program_with_binary
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-program-with-binary")
add_test(NAME piglit_cl_api_create_program_with_source
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-program-with-source")
add_test(NAME piglit_cl_api_create_sampler
         COMMAND "${TS_BUILDDIR}/bin/cl-api-create-sampler")
add_test(NAME piglit_cl_api_enqueue_copy_buffer
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-copy-buffer")
add_test(NAME piglit_cl_api_enqueue_copy_buffer_rect
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-copy-buffer-rect")
add_test(NAME piglit_cl_api_enqueue_fill_buffer
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-fill-buffer")
add_test(NAME piglit_cl_api_enqueue_fill_image
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-fill-image")
add_test(NAME piglit_cl_api_enqueue_map_buffer
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-map-buffer")
add_test(NAME piglit_cl_api_enqueue_migrate_mem_objects
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-migrate-mem-objects")
add_test(NAME piglit_cl_api_enqueue_read_write_buffer
         COMMAND "${TS_BUILDDIR}/bin/cl-api-enqueue-read_write-buffer")
add_test(NAME piglit_cl_api_get_command_queue_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-command-queue-info")
add_test(NAME piglit_cl_api_get_context_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-context-info")
add_test(NAME piglit_cl_api_get_device_ids
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-device-ids")
add_test(NAME piglit_cl_api_get_device_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-device-info")
add_test(NAME piglit_cl_api_get_event_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-event-info")
add_test(NAME piglit_cl_api_get_image_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-image-info")
add_test(NAME piglit_cl_api_get_kernel_arg_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-kernel-arg-info")
add_test(NAME piglit_cl_api_get_kernel_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-kernel-info")
add_test(NAME piglit_cl_api_get_kernel_work_group_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-kernel-work-group-info")
add_test(NAME piglit_cl_api_get_mem_object_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-mem-object-info")
add_test(NAME piglit_cl_api_get_platform_ids
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-platform-ids")
add_test(NAME piglit_cl_api_get_platform_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-platform-info")
add_test(NAME piglit_cl_api_get_program_build_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-program-build-info")
add_test(NAME piglit_cl_api_get_program_info
         COMMAND "${TS_BUILDDIR}/bin/cl-api-get-program-info")
add_test(NAME piglit_cl_api_link_program
         COMMAND "${TS_BUILDDIR}/bin/cl-api-link-program")
add_test(NAME piglit_cl_api_retain_release_command_queue
         COMMAND "${TS_BUILDDIR}/bin/cl-api-retain_release-command-queue")
add_test(NAME piglit_cl_api_retain_release_context
         COMMAND "${TS_BUILDDIR}/bin/cl-api-retain_release-context")

# disable retain-release event test.
# has two issues:
#   1) has a race condition (events sometime finish before
#      it asks for status, sometimes after)
#   2) it thinks CL_EVENT_REFERENCE_COUNT should be 1 after
#      an API call clEnqueueXYZ(... , &event) - but pocl inits it to 2
#      (one for pocl, one for user)
#add_test(NAME piglit_cl_api_retain_release_event
         #COMMAND "${TS_BUILDDIR}/bin/cl-api-retain_release-event")
add_test(NAME piglit_cl_api_retain_release_kernel
         COMMAND "${TS_BUILDDIR}/bin/cl-api-retain_release-kernel")
add_test(NAME piglit_cl_api_retain_release_mem_object
         COMMAND "${TS_BUILDDIR}/bin/cl-api-retain_release-mem-object")
add_test(NAME piglit_cl_api_retain_release_program
         COMMAND "${TS_BUILDDIR}/bin/cl-api-retain_release-program")
add_test(NAME piglit_cl_api_set_kernel_arg
         COMMAND "${TS_BUILDDIR}/bin/cl-api-set-kernel-arg")
add_test(NAME piglit_cl_api_unload_compiler
         COMMAND "${TS_BUILDDIR}/bin/cl-api-unload-compiler")
add_test(NAME piglit_cl_custom_buffer_flags
         COMMAND "${TS_BUILDDIR}/bin/cl-custom-buffer-flags")
add_test(NAME piglit_cl_custom_flush_after_enqueue_kernel
         COMMAND "${TS_BUILDDIR}/bin/cl-custom-flush-after-enqueue-kernel")
add_test(NAME piglit_cl_custom_r600_create_release_buffer_bug
         COMMAND "${TS_BUILDDIR}/bin/cl-custom-r600-create-release-buffer-bug")
add_test(NAME piglit_cl_custom_run_simple_kernel
         COMMAND "${TS_BUILDDIR}/bin/cl-custom-run-simple-kernel")
add_test(NAME piglit_cl_custom_use_sub_buffer_in_kernel
         COMMAND "${TS_BUILDDIR}/bin/cl-custom-use-sub-buffer-in-kernel")
add_test(NAME piglit_cl_program_bitcoin_phatk
         COMMAND "${TS_BUILDDIR}/bin/cl-program-bitcoin-phatk")
add_test(NAME piglit_cl_program_max_work_item_sizes
         COMMAND "${TS_BUILDDIR}/bin/cl-program-max-work-item-sizes")


set_tests_properties(
  piglit_cl_api_build_program
  piglit_cl_api_compile_program
  piglit_cl_api_create_buffer
  piglit_cl_api_create_command_queue
  piglit_cl_api_create_context
  piglit_cl_api_create_context_from_type
  piglit_cl_api_create_image
  piglit_cl_api_create_kernel
  piglit_cl_api_create_kernels_in_program
  piglit_cl_api_create_program_with_binary
  piglit_cl_api_create_program_with_source
  piglit_cl_api_create_sampler
  piglit_cl_api_enqueue_copy_buffer
  piglit_cl_api_enqueue_copy_buffer_rect
  piglit_cl_api_enqueue_fill_buffer
  piglit_cl_api_enqueue_fill_image
  piglit_cl_api_enqueue_map_buffer
  piglit_cl_api_enqueue_migrate_mem_objects
  piglit_cl_api_enqueue_read_write_buffer
  piglit_cl_api_get_command_queue_info
  piglit_cl_api_get_context_info
  piglit_cl_api_get_device_ids
  piglit_cl_api_get_device_info
  piglit_cl_api_get_event_info
  piglit_cl_api_get_image_info
  piglit_cl_api_get_kernel_arg_info
  piglit_cl_api_get_kernel_info
  piglit_cl_api_get_kernel_work_group_info
  piglit_cl_api_get_mem_object_info
  piglit_cl_api_get_platform_ids
  piglit_cl_api_get_platform_info
  piglit_cl_api_get_program_build_info
  piglit_cl_api_get_program_info
  piglit_cl_api_link_program
  piglit_cl_api_retain_release_command_queue
  piglit_cl_api_retain_release_context
#  piglit_cl_api_retain_release_event
  piglit_cl_api_retain_release_kernel
  piglit_cl_api_retain_release_mem_object
  piglit_cl_api_retain_release_program
  piglit_cl_api_set_kernel_arg
  piglit_cl_api_unload_compiler
  piglit_cl_custom_buffer_flags
  piglit_cl_custom_flush_after_enqueue_kernel
  piglit_cl_custom_r600_create_release_buffer_bug
  piglit_cl_custom_run_simple_kernel
  piglit_cl_custom_use_sub_buffer_in_kernel
  piglit_cl_program_bitcoin_phatk
  piglit_cl_program_max_work_item_sizes

  PROPERTIES
    PASS_REGULAR_EXPRESSION "pass"
    FAIL_REGULAR_EXPRESSION "fail"
    LABELS "piglit")

set_tests_properties(
  piglit_cl_api_create_program_with_binary

  PROPERTIES
    WILL_FAIL 1)

endif()
