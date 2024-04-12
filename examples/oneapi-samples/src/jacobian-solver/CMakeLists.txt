# set(CMAKE_CXX_FLAGS "")

add_executable(jacobian_solver_multidevice_sycl src/jacobian_gpu_sycl_multidevice.cpp)
target_link_libraries (jacobian_solver_multidevice_sycl PRIVATE ${SYCL_LIBRARIES} -fsycl)

target_compile_options("jacobian_solver_multidevice_sycl" PUBLIC "-fsycl" "-fsycl-targets=spir64")

add_test (NAME oneapi_jacobian_solver_multidevice_sycl-cpu
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/jacobian_solver_multidevice_sycl
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ltbb")

set_tests_properties("oneapi_jacobian_solver_multidevice_sycl-cpu"
  PROPERTIES
  PASS_REGULAR_EXPRESSION "OK"
  FAIL_REGULAR_EXPRESSION "FAIL"
  ENVIRONMENT "POCL_DEVICES='cpu cpu cpu cpu cpu cpu cpu cpu';POCLD_COARSE_GRAIN_SVM=1;POCL_CPU_VENDOR_ID_OVERRIDE=32902;POCL_DRIVER_VERSION_OVERRIDE=2023.16.7.0.21_160000;ONEAPI_DEVICE_SELECTOR='opencl:cpu'"
  )

add_test (NAME oneapi_jacobian_solver_multidevice_sycl-remote
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}/../../../.." examples/oneapi-samples/src/oneapi-samples-build/jacobian-solver/jacobian_solver_multidevice_sycl
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties("oneapi_jacobian_solver_multidevice_sycl-remote"
  PROPERTIES
  PASS_REGULAR_EXPRESSION "OK"
  FAIL_REGULAR_EXPRESSION "FAIL"
  ENVIRONMENT "POCL_DEVICES='remote remote';POCLD_COARSE_GRAIN_SVM=1;POCL_CPU_VENDOR_ID_OVERRIDE=32902;POCL_DRIVER_VERSION_OVERRIDE=2023.16.7.0.21_160000;ONEAPI_DEVICE_SELECTOR='opencl:cpu'"
)

# We use the barrier etc.
set(CMAKE_CXX_STANDARD 20)
