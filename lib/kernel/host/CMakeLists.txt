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

list(APPEND KERNEL_SOURCES "mem_fence.c")

if(HOST_DEVICE_CL_VERSION GREATER 199)
if(X86_64 OR I386)
  if(LLVM_3_6)
    message(STATUS "OpenCL 2.0 atomics only works with LLVM >= 3.7")
  elseif(LLVM_OLDER_THAN_3_9)
    list(APPEND KERNEL_SOURCES svm_atomics_x86_64.ll svm_atomics.cl)
  elseif(LLVM_3_9 AND POCL_USE_FAKE_ADDR_SPACE_IDS)
    list(APPEND KERNEL_SOURCES svm_atomics_x86_64_llvm3_9.ll svm_atomics.cl)
  else()
    list(APPEND KERNEL_SOURCES svm_atomics_x86_64_no_fake_asids.ll svm_atomics.cl)
  endif()
elseif(MIPS)
  message(STATUS "OpenCL 2.0 atomics are currently broken on MIPS")
else()
  message(STATUS "Using generic OpenCL 2.0 atomics. Might or might not break your build.")
  list(APPEND KERNEL_SOURCES svm_atomics_host.cl svm_atomics.cl)
endif()
endif()

set(KERNEL_CL_FLAGS "-Wall" "-Wno-unused-local-typedef" "-Xclang" "-cl-std=CL${HOST_DEVICE_CL_STD}" "-D__OPENCL_C_VERSION__=${HOST_DEVICE_CL_VERSION}" ${KERNEL_CL_FLAGS})
separate_arguments(HOST_CLANG_FLAGS)
separate_arguments(HOST_LLC_FLAGS)
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${HOST_DEVICE_CL_VERSION} ${HOST_DEVICE_EXTENSION_DEFINES}")
separate_arguments(DEVICE_CL_FLAGS)


function(x86_distro_variant_to_flags VARIANT OUT_LLC_FLAGS OUT_CLANG_FLAGS)

  if("${VARIANT}" STREQUAL "sse2")
    set(CLANG_F "${CLANG_MARCH_FLAG}athlon64")
    set(LLC_F "-mcpu=athlon64")

  elseif("${VARIANT}" STREQUAL "ssse3")
    set(CLANG_F "${CLANG_MARCH_FLAG}core2")
    set(LLC_F "-mcpu=core2")

  elseif("${VARIANT}" STREQUAL "sse41")
    set(CLANG_F "${CLANG_MARCH_FLAG}penryn")
    set(LLC_F "-mcpu=penryn")

  elseif("${VARIANT}" STREQUAL "avx")
    set(CLANG_F "${CLANG_MARCH_FLAG}sandybridge")
    set(LLC_F "-mcpu=sandybridge")

  elseif("${VARIANT}" STREQUAL "avx_f16c")
    set(CLANG_F "${CLANG_MARCH_FLAG}ivybridge")
    set(LLC_F "-mcpu=ivybridge")

  elseif("${VARIANT}" STREQUAL "avx_fma4")
    set(CLANG_F "${CLANG_MARCH_FLAG}bdver1")
    set(LLC_F "-mcpu=bdver1")

  elseif("${VARIANT}" STREQUAL "avx2")
    set(CLANG_F "${CLANG_MARCH_FLAG}haswell")
    set(LLC_F "-mcpu=haswell")

  elseif("${VARIANT}" STREQUAL "avx512")
    set(CLANG_F "${CLANG_MARCH_FLAG}skylake")
    set(LLC_F "-mcpu=skylake")

  else()
    set(CLANG_F "${CLANG_MARCH_FLAG}${VARIANT}")
    set(LLC_F "-mcpu=${VARIANT}")

  endif()

  set(${OUT_LLC_FLAGS} "${LLC_F}" PARENT_SCOPE)
  set(${OUT_CLANG_FLAGS} "${CLANG_F}" PARENT_SCOPE)
endfunction()

###############################################################################

foreach(CPU_VARIANT IN LISTS KERNELLIB_HOST_CPU_VARIANTS)

if(CPU_VARIANT MATCHES "native")
  set(VARIANT "${LLC_HOST_CPU}")
else()
  set(VARIANT "${CPU_VARIANT}")
endif()

if(X86_64 OR I386)
  x86_distro_variant_to_flags("${VARIANT}" LLC_CPUFLAGS CLANG_CPUFLAGS)
else()
  set(CLANG_CPUFLAGS "${CLANG_MARCH_FLAG}${VARIANT}")
  set(LLC_CPUFLAGS "-mcpu=${VARIANT}")
endif()

separate_arguments(CLANG_CPUFLAGS)
separate_arguments(LLC_CPUFLAGS)
set(CLANG_FLAGS ${HOST_CLANG_FLAGS} ${CLANG_CPUFLAGS} "-emit-llvm" "-ffp-contract=off")

if(POCL_USE_FAKE_ADDR_SPACE_IDS)
list(APPEND CLANG_FLAGS "-Xclang" "-ffake-address-space-map" "-DPOCL_USE_FAKE_ADDR_SPACE_IDS")
endif()

set(LLC_FLAGS ${HOST_LLC_FLAGS} ${LLC_CPUFLAGS})

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
