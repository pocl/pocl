# set(CMAKE_CXX_FLAGS "")

find_package(TBB QUIET PATHS /opt/intel/oneapi/tbb/latest/lib/cmake/tbb)

if(NOT TARGET TBB::tbb)
  message(WARNING "disabling jacobian-solver multidevice, requires TBB library (not found)")
  return()
endif()

add_executable(jacobian_solver_multidevice_sycl src/jacobian_gpu_sycl_multidevice.cpp)
target_link_libraries (jacobian_solver_multidevice_sycl PRIVATE ${SYCL_LIBRARIES} TBB::tbb)
target_compile_options(jacobian_solver_multidevice_sycl PRIVATE "-fsycl" "-fsycl-targets=spir64")

add_test (NAME oneapi_jacobian_solver_multidevice_sycl
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/jacobian_solver_multidevice_sycl
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties("oneapi_jacobian_solver_multidevice_sycl"
  PROPERTIES
  PASS_REGULAR_EXPRESSION "OK"
  FAIL_REGULAR_EXPRESSION "FAIL"
  LABELS "multidevice_local")

# TODO there is way to run a test through launcher since cmake 3.29,
# but it's only build-time configurable, not runtime configurable
add_test (NAME oneapi_jacobian_solver_multidevice_sycl-remote
  COMMAND "${CMAKE_SOURCE_DIR}/tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" $<TARGET_FILE:jacobian_solver_multidevice_sycl>
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties("oneapi_jacobian_solver_multidevice_sycl-remote"
  PROPERTIES
  PASS_REGULAR_EXPRESSION "OK"
  FAIL_REGULAR_EXPRESSION "FAIL"
  LABELS "multidevice_remote")

# Disabled until the PoCL-CPU execution issue with the reduction loop is resolved.
#add_executable(jacobian_solver_multidevice_openmp src/jacobian_gpu_openmp_multidevice.cpp)

#target_compile_options("jacobian_solver_multidevice_openmp" PUBLIC "-qopenmp" "-fopenmp-targets=spir64")
#target_link_libraries (jacobian_solver_multidevice_openmp PRIVATE -fopenmp -lomptarget)

#add_test (NAME oneapi_jacobian_solver_multidevice_openmp
#  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/jacobian_solver_multidevice_openmp
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# We use the barrier etc.
set(CMAKE_CXX_STANDARD 20)
