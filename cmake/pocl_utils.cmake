#=============================================================================
#   CMake utility functions.
#
#   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy
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

# Ceases the configuration if symbolic links can't be created on the system.
function(pocl_assert_symlinks_works)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/test_symlink_target
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_BINARY_DIR}/test_symlink_target
    ${CMAKE_BINARY_DIR}/test_symlink_itself
    RESULTS_VARIABLE ECS
    COMMAND_ECHO STDOUT)

  list(GET ECS 0 TOUCH_EC)
  if (TOUCH_EC)
    message(FATAL_ERROR "Can't create files in the build directory!")
  endif()

  list(GET ECS 1 MKLINK_EC)
  if(MKLINK_EC)
    if(WIN32)
      message(NOTICE "Symbolic links requires elevated permissions or developer mode enabled on Windows.")
    endif()
    message(FATAL_ERROR "Can't create symbolic links!")
  endif()
  message(STATUS "symlinks works.")
endfunction()

# Creates symlink named OpenCL.dll (and libhwloc dll) in the given directory
# to the actually built libpocl(=OpenCL.dll) and libhwloc. This is needed so the
# internal tests and examples, outside the runtime library directory,
# link to the PoCL one rather than to a system one.  Adjusting PATH
# environment doesn't cut because the system libraries are considered
# before the paths in PATH (and Windows does not have equivalent of
# LD_LIBRARY_PATH nor runpaths).
#
# Example usage:
#
#   add_symlink_to_built_opencl_dynlib_in_dir("C:\pocl-build\bin" CMake-Target ...)
#
# Providing a Target ensures that the the symlink is created. If a executable X
# is not passed to this function, running a rule - for example, 'make X' -
# may not create the symlink.
function(add_symlink_to_built_opencl_dynlib_in_dir SYMLINK_DIR DEPENDENCY_TARGET)
  # TODO: Skip the symlink creation if the current directory is the one where
  #       the OpenCL.dll is.

  # Skip if PoCL is not building OpenCL runtime library on Windows.
  if(NOT WIN32 AND NOT BUILD_SHARED_LIBS OR ENABLE_ICD)
    return()
  endif()

  # Name the custom target so that the symlink is created once per directory.
  set(SYMLINK_TGT_NAME ${SYMLINK_DIR}/symlink_target)
  string(REPLACE "/" "_" SYMLINK_TGT_NAME "${SYMLINK_TGT_NAME}")
  string(REPLACE "\\" "_" SYMLINK_TGT_NAME "${SYMLINK_TGT_NAME}")
  string(REPLACE ":" "_" SYMLINK_TGT_NAME "${SYMLINK_TGT_NAME}")
  string(REPLACE "-" "_" SYMLINK_TGT_NAME "${SYMLINK_TGT_NAME}")

  set(SYMLINK_TGT_DIR "${SYMLINK_DIR}")
  get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(is_multi_config)
    set(SYMLINK_TGT_DIR "${SYMLINK_DIR}/$<CONFIG>")
  endif()

  #message(STATUS "Add custom target NAMED: ${SYMLINK_TGT_NAME}")
  if (NOT TARGET ${SYMLINK_TGT_NAME})
    #message(STATUS "Add custom command with SYMLINK_DIR: ${SYMLINK_DIR}")
    add_custom_command(
      OUTPUT "${SYMLINK_TGT_DIR}/OpenCL.dll"
	  DEPENDS ${POCL_LIBRARY_NAME}
      COMMAND ${CMAKE_COMMAND} -E create_symlink
      "$<TARGET_FILE:${POCL_LIBRARY_NAME}>"  "${SYMLINK_TGT_DIR}/OpenCL.dll"
      COMMENT "CTS: create symbolic link to PoCL-built OpenCL.dll")
    set(CUST_TGT_DEPENDS "${SYMLINK_TGT_DIR}/OpenCL.dll")

    if(ENABLE_HWLOC AND TARGET PkgConfig::HWLOC)
      get_target_property(HWLOC_DLL_FULL_PATH PkgConfig::HWLOC IMPORTED_LOCATION)
      get_filename_component(HWLOC_DLL_NAME_ONLY "${HWLOC_DLL_FULL_PATH}" NAME)
      if(HWLOC_DLL_FULL_PATH AND HWLOC_DLL_NAME_ONLY)
        add_custom_command(OUTPUT "${SYMLINK_TGT_DIR}/${HWLOC_DLL_NAME_ONLY}"
	  DEPENDS ${POCL_LIBRARY_NAME}
          COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${HWLOC_DLL_FULL_PATH}"  "${SYMLINK_TGT_DIR}/${HWLOC_DLL_NAME_ONLY}"
          COMMENT "CTS: create symbolic link to ${HWLOC_DLL_NAME_ONLY}")
        list(APPEND CUST_TGT_DEPENDS "${SYMLINK_TGT_DIR}/${HWLOC_DLL_NAME_ONLY}")
      endif()
    endif()

    add_custom_target(${SYMLINK_TGT_NAME} DEPENDS ${CUST_TGT_DEPENDS})
  endif()

  #message(STATUS "adding dep TARGET: ${TARGETS} DEP: ${SYMLINK_TGT_NAME}")
  add_dependencies(${DEPENDENCY_TARGET} ${SYMLINK_TGT_NAME})
endfunction()

# creates a symlink in the current binary directory.
function(add_symlink_to_built_opencl_dynlib TARGETS)
  add_symlink_to_built_opencl_dynlib_in_dir("${CMAKE_CURRENT_BINARY_DIR}" ${TARGETS})
endfunction()
