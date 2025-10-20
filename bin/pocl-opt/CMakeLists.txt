#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2025 pocl developers
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

add_mlir_tool(pocl-opt
    pocl-opt.cpp
)

llvm_update_compile_flags(pocl-opt)

target_include_directories(pocl-opt SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS})
target_include_directories(pocl-opt PRIVATE "${CMAKE_SOURCE_DIR}/lib/mliropencl/include")
target_include_directories(pocl-opt PRIVATE "${CMAKE_BINARY_DIR}/lib/mliropencl/include")

target_link_libraries(pocl-opt
    PRIVATE
    MLIR
    PoclDialect
    "$<TARGET_OBJECTS:mlirpasses>"
)
