#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2024 pocl developers
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
include(ExternalProject)

find_program(TREE_SITTER_CLI tree-sitter DOC "Tree-sitter CLI" REQUIRED)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
ExternalProject_Add(tree_sitter
        GIT_REPOSITORY https://github.com/lefp/tree-sitter-opencl.git
        # Generate Cmake config
        PATCH_COMMAND ${TREE_SITTER_CLI} init
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DBUILD_SHARED_LIBS=OFF
)

add_library(pocl_tree_sitter_utils OBJECT pocl_tree_sitter_utils.c
        pocl_tree_sitter_utils.h)
target_include_directories(pocl_tree_sitter_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EXTERNAL_INSTALL_LOCATION}/include)
target_link_directories(pocl_tree_sitter_utils PUBLIC ${EXTERNAL_INSTALL_LOCATION}/lib)
add_dependencies(pocl_tree_sitter_utils tree_sitter)

target_link_libraries(pocl_tree_sitter_utils PUBLIC tree-sitter-opencl ${TSLIB})
