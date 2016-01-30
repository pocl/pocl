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

if(USE_VECMATHLIB)
  set(KERNEL_SOURCES ${SOURCES_WITH_VML})
else()
  set(KERNEL_SOURCES ${SOURCES_WITHOUT_VML})
endif()

if(LLVM_3_6)
  message(STATUS "OpenCL 2.0 atomics don't work with LLVM 3.6")
else()
  if(X86_64)
    list(APPEND KERNEL_SOURCES svm_atomics_x86_64.ll svm_atomics.cl)
  elseif(MIPS)
    message(STATUS "OpenCL 2.0 atomics are currently broken on MIPS")
  else()
    message(STATUS "Using generic OpenCL 2.0 atomics. Might or might not break your build.")
    list(APPEND KERNEL_SOURCES svm_atomics_host.cl svm_atomics.cl)
  endif()
endif()

separate_arguments(HOST_CLANG_FLAGS)
separate_arguments(HOST_LLC_FLAGS)
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${HOST_DEVICE_CL_VERSION} ${HOST_DEVICE_EXTENSION_DEFINES}")
separate_arguments(DEVICE_CL_FLAGS)

foreach(CPU_VARIANT IN LISTS KERNELLIB_HOST_CPU_VARIANTS)

if(CPU_VARIANT MATCHES "native")
  set(VARIANT "${LLC_HOST_CPU}")
else()
  set(VARIANT "${CPU_VARIANT}")
endif()

set(CLANG_CPUFLAG "${CLANG_MARCH_FLAG}${VARIANT}")
set(LLC_CPUFLAG "-mcpu=${VARIANT}")

set(CLANG_FLAGS ${HOST_CLANG_FLAGS} ${CLANG_CPUFLAG} "-Xclang" "-ffake-address-space-map" "-emit-llvm" "-ffp-contract=off")

set(LLC_FLAGS ${HOST_LLC_FLAGS} ${LLC_CPUFLAG})

# KERNEL_TARGET = @OCL_KERNEL_TARGET@
make_kernel_bc(KERNEL_BC "${OCL_KERNEL_TARGET}-${VARIANT}" "${VARIANT}" ${KERNEL_SOURCES})

# just debug
message(STATUS "Host Kernel BC for \"${VARIANT}\": ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_host_${VARIANT}" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_host_${VARIANT}")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")

endforeach()
