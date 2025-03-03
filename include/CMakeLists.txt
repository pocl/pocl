#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

if(INSTALL_OPENCL_HEADERS)
  add_subdirectory("CL")
endif()

set(PRIVATE_HEADERS _enable_all_exts.h _builtin_renames.h
                    _kernel.h _clang_opencl.h opencl-c.h opencl-c-base.h
                    _kernel_c.h _kernel_constants.h
                    pocl_types.h pocl_device.h pocl.h pocl_spir.h
                    pocl_image_types.h)

cpack_add_component("dev")


if(ENABLE_HEADER_BUNDLING)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/HeaderBundle")

  function(create_bundled_header HEADER OUTPUT_VAR)
    string(REPLACE "." "_" HEADER_NAME ${HEADER})
    string(REPLACE "-" "_" HEADER_NAME ${HEADER_NAME})
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/HeaderBundle/${HEADER}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}"
        COMMAND "${CMAKE_COMMAND}"
        "-Dname=${HEADER_NAME}"
        "-Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}"
        "-Doutput=${CMAKE_BINARY_DIR}/HeaderBundle/${HEADER}"
        -P "${CMAKE_SOURCE_DIR}/cmake/generate_c_header_file.cmake"
        COMMENT "Generating bundling form of header ${HEADER}"
        VERBATIM)
    set(${OUTPUT_VAR} "${CMAKE_BINARY_DIR}/HeaderBundle/${HEADER}" PARENT_SCOPE)
  endfunction()

  foreach(HDR_FILE IN ITEMS _clang_opencl.h  opencl-c-base.h  opencl-c.h)
    create_bundled_header(${HDR_FILE} TEMP)
    list(APPEND HEADER_BUNDLE_FILES "${TEMP}")
  endforeach()

  add_custom_target(regenerate_bundled_headers ALL DEPENDS ${HEADER_BUNDLE_FILES})
endif()

# this can still be required by some drivers, even if bundling is enabled
install(FILES ${PRIVATE_HEADERS}
        DESTINATION ${POCL_INSTALL_PRIVATE_HEADER_DIR_REL} COMPONENT "dev")
