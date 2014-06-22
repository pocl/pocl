

add_test("pocl_version_check" "runtime/test_version")
set_tests_properties("pocl_version_check"
  PROPERTIES
  PASS_REGULAR_EXPRESSION "pthread")

#######################################################################

add_subdirectory("kernel")
if(HAVE_OPENCL_HPP)
  add_subdirectory("regression")
else()
  message(STATUS "cl.hpp unavailable, skipping regression tests")
endif()
add_subdirectory("runtime")
add_subdirectory("workgroup")
if(ENABLE_TCE)
  #add_subdirectory("tce")
endif()
if(ENABLE_SPU)
  #add_subdirectory("cell")
endif()

