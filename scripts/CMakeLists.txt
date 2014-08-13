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

if(UNIX)
  # build-dir script
  get_target_property(LLVMOPENCL_LOCATION "llvmopencl" LOCATION)
  set(KERNEL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(ADD_INCLUDE "-I${CMAKE_BINARY_DIR}")
  set(FULL_TARGET_DIR "${CMAKE_BINARY_DIR}/lib/kernel/\${target_dir}")
  configure_file("pocl-standalone.in.cmake" "pocl-standalone" ESCAPE_QUOTES @ONLY)

  # install script
  set(LLVMOPENCL_LOCATION "${POCL_INSTALL_PRIVATE_LIBDIR}/llvmopencl.so")
  set(KERNEL_INCLUDE_DIR "${POCL_INSTALL_PRIVATE_HEADER_DIR}")
  set(ADD_INCLUDE "")
  set(FULL_TARGET_DIR "${POCL_INSTALL_PRIVATE_DATADIR}")
  configure_file("pocl-standalone.in.cmake" "pocl-standalone.install" ESCAPE_QUOTES @ONLY)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pocl-standalone.install" RENAME "pocl-standalone"
        DESTINATION ${POCL_INSTALL_PUBLIC_BINDIR})
endif()
