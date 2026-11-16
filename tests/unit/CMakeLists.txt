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
  add_symlink_to_built_opencl_dynlib(${BASENAME})

  target_link_libraries(${BASENAME}
    PRIVATE ${POCL_LIBRARY_NAME} ${POCLU_LINK_OPTIONS})

  target_include_directories(${BASENAME}
    PRIVATE $<TARGET_PROPERTY:${POCL_LIBRARY_NAME},INCLUDE_DIRECTORIES>)

  # Set runpath, In case the PoCL library is named as libOpenCL.so, to
  # avoid picking up another OpenCL library in the system.
  set_target_properties(${BASENAME} PROPERTIES BUILD_RPATH
    $<TARGET_FILE_DIR:${POCL_LIBRARY_NAME}>)

  add_test(NAME "unit/${BASENAME}"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME})

endfunction()

add_unit_test(test_fs.cc)
add_unit_test(test_runcmds.cc)

if(ENABLE_HOST_CPU_DEVICES AND ENABLE_LLVM AND KERNELLIB_HOST_DISTRO_VARIANTS)
  add_unit_test(test_distro_cpu.c)
  if(X86_64)
    add_unit_test(test_distro_cpu_features.cc)
    target_include_directories(test_distro_cpu_features PRIVATE ${LLVM_INCLUDE_DIRS})
    if(POCL_LLVM_LINK_TARGETS)
      target_link_libraries(test_distro_cpu_features PRIVATE ${POCL_LLVM_LINK_TARGETS} ${LLVM_SYSLIBS})
    else()
      target_link_libraries(test_distro_cpu_features PRIVATE ${LLVM_LINK_LIBRARIES} ${LLVM_SYSLIBS})
    endif()
    add_test(NAME "unit/test_distro_cpu_incompatible_compile"
      COMMAND ${CMAKE_COMMAND} -E env POCL_LLVM_CPU_NAME=core2 POCL_KERNELLIB_NAME=avx2
        $<TARGET_FILE:poclcc> -o distro-cpu-half.pocl
        "${CMAKE_CURRENT_SOURCE_DIR}/../compiler_unit/distro-cpu-half.cl")
    foreach(MODE library override both incompatible baseline znver3 znver4 unknown)
      add_test(NAME "unit/test_distro_cpu_${MODE}"
        COMMAND test_distro_cpu ${MODE})
    endforeach()
  endif()
endif()
