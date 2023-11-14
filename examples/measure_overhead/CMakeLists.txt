#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2019 pocl developers
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

add_compile_options(${OPENCL_CFLAGS})

add_executable("measure_round_trip_overhead" measure_round_trip_overhead.cc common.cc)
add_executable("measure_migration_overhead" measure_migration_overhead.cc common.cc)
add_executable("measure_distributed_matmul" measure_distributed_matmul.cc common.cc)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(TARGET measure_round_trip_overhead PROPERTY CXX_STANDARD 17)
set_property(TARGET measure_migration_overhead PROPERTY CXX_STANDARD 17)
set_property(TARGET measure_distributed_matmul PROPERTY CXX_STANDARD 17)

target_link_libraries("measure_round_trip_overhead" ${POCLU_LINK_OPTIONS})
target_link_libraries("measure_migration_overhead" ${POCLU_LINK_OPTIONS})
target_link_libraries("measure_distributed_matmul" ${POCLU_LINK_OPTIONS})
