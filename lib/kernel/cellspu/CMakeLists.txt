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

include("bitcode_rules")

# Use TARGET flags:
#CLANG_FLAGS = @TARGET_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off
separate_arguments(CELL_TARGET_CLANG_FLAGS)
set(CLANG_FLAGS ${CELL_TARGET_CLANG_FLAGS} "-Xclang" "-ffake-address-space-map" "-emit-llvm" "-ffp-contract=off")

#LLC_FLAGS   = @TARGET_LLC_FLAGS@
separate_arguments(CELL_TARGET_LLC_FLAGS)
set(LLC_FLAGS ${CELL_TARGET_LLC_FLAGS})

# TODO LLC_flags is used by kernel.bc target, but ld is unused
#LD_FLAGS    = @TARGET_LD_FLAGS@

#KERNEL_TARGET = tce (WRONG)
make_kernel_bc(KERNEL_BC "cellspu" ${SOURCES_WITHOUT_VML})

# just debug
message(STATUS "Cell SPU Kernel BC: ${KERNEL_BC}")

# a target is needed...
add_custom_target("kernel_cell" ALL DEPENDS ${KERNEL_BC})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")
