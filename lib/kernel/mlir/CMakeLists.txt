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

include("bitcode_rules")


set(MLIR_KERNELLIB_SOURCES
"mlir/get_global_id.cl"
"mlir/get_global_offset.cl"
"mlir/get_global_size.c"
"mlir/math.mlir"
)

set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=120 -DPOCL_DEVICE_ADDRESS_BITS=64 -fclangir -emit-cir")
separate_arguments(DEVICE_CL_FLAGS)

set(KERNEL_CL_FLAGS "-D__OPENCL_C_VERSION__=120" )

set(KERNEL_C_FLAGS -xc -std=c11 -D__CBUILD__ -fno-math-errno -fno-stack-protector -fPIC )

make_kernel_mlir(KERNEL_MLIR "mlirbc" "mlir" 0 0 0 ${MLIR_KERNELLIB_SOURCES})

add_custom_target("kernel_mlir" DEPENDS ${KERNEL_MLIR})

list(APPEND KERNEL_TARGET_LIST "kernel_mlir")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_MLIR}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")
