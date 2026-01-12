#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2025 Topi Leppänen / Tampere University
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

add_mlir_dialect_library(PoclDialect OBJECT
                        PoclDialect.cpp
                        DEPENDS
                        MLIRPoclOpsIncGen
                        )

if(NOT LLVM_HAS_RTTI)
  if(MSVC)
    target_compile_options(obj.PoclDialect PRIVATE "/GR-")
  else()
    target_compile_options(obj.PoclDialect PRIVATE "-fno-rtti")
  endif()
endif()

target_include_directories(PoclDialect PRIVATE ${MLIR_INCLUDE_DIRS})
target_include_directories(PoclDialect PRIVATE "${CMAKE_SOURCE_DIR}/lib/mliropencl/include")
target_include_directories(PoclDialect PRIVATE "${CMAKE_BINARY_DIR}/lib/mliropencl/include")

install(TARGETS PoclDialect
        LIBRARY DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR_REL}
        COMPONENT "lib"
        RUNTIME DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR_REL}
        COMPONENT "lib")
