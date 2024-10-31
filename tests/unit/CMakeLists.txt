#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2024 PoCL Developers
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

function(add_unit_test MAIN_SOURCE)
  set(multiValueArgs COMPILE_DEFINITIONS)
  cmake_parse_arguments(UNIT "" "" "${multiValueArgs}" ${ARGN})

  get_filename_component(BASENAME ${MAIN_SOURCE} NAME_WLE)
  add_executable(${BASENAME} ${MAIN_SOURCE})
  target_compile_definitions(${BASENAME} PRIVATE ${UNIT_COMPILE_DEFINITIONS})

  target_link_libraries(${BASENAME} PRIVATE ${POCL_LIBRARY_NAME})
  if(SANITIZER_OPTIONS)
    target_link_libraries(${BASENAME} PRIVATE ${SANITIZER_LIBS})
  endif()

  target_include_directories(${BASENAME}
    PRIVATE $<TARGET_PROPERTY:${POCL_LIBRARY_NAME},INCLUDE_DIRECTORIES>)

  # Set runpath, In case the PoCL library is named as libOpenCL.so, to
  # avoid picking up another OpenCL library in the system.
  set_target_properties(${BASENAME} PROPERTIES BUILD_RPATH
    $<TARGET_FILE_DIR:${POCL_LIBRARY_NAME}>)

  add_test(NAME "unit/${BASENAME}"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME})

  set_tests_properties("unit/${BASENAME}"
      PROPERTIES LABELS "internal")
endfunction()

add_unit_test(test_fs.cc)
