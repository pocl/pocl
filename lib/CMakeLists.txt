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

separate_arguments(HOST_CLANG_FLAGS)
separate_arguments(HOST_LLC_FLAGS)

if (OCS_AVAILABLE)
  add_subdirectory("kernel")
  add_subdirectory("llvmopencl")
endif()

set(SANITIZER_OPTIONS "")

if(ENABLE_ASAN)
  if("${CMAKE_C_COMPILER_VERSION}" VERSION_LESS "6.0.0")
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address")
  else()
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address" "-fsanitize-recover=address")
  endif()
  list(APPEND SANITIZER_LIBS "asan")
endif()

if(ENABLE_LSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=leak")
  list(APPEND SANITIZER_LIBS "lsan")
endif()

if(ENABLE_TSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=thread")
  list(APPEND SANITIZER_LIBS "tsan")
endif()

if(ENABLE_UBSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=undefined")
  list(APPEND SANITIZER_LIBS "ubsan")
endif()

if(SANITIZER_OPTIONS)
  list(APPEND SANITIZER_OPTIONS "-fno-omit-frame-pointer")
  string(REPLACE ";" " " SANITIZER_OPTIONS_STR "${SANITIZER_OPTIONS}")
endif()

add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

add_subdirectory("CL")

#############################################################

# must be after adding lib/CL subdir b/c of libpocl target,
# but before poclu & examples & tests (they need these variables)

if(ENABLE_ICD)

  # -pthread makes gdb happier when debugging tests
  set(OPENCL_LIBS "${PTHREAD_LIBRARY};${OPENCL_LIBRARIES}")

else()

  # when we link pocl directly as libOpenCL.so,
  # llvm+clang libs are a transitive dependency of
  # executables (e.g. poclcc -> libOpenCL -> libLLVM)
  # and this must be specified on new systems,
  # otherwise linking fails.
  set(OPENCL_LIBS "${PTHREAD_LIBRARY};${POCL_LIBRARY_NAME};${POCL_TRANSITIVE_LIBS}")

endif()

set(OPENCL_CFLAGS "")

if(SANITIZER_OPTIONS)
  list(INSERT OPENCL_LIBS 0 ${SANITIZER_LIBS})
  set(OPENCL_CFLAGS "${OPENCL_CFLAGS} ${SANITIZER_OPTIONS_STR}")
endif()

set(OPENCL_CFLAGS_STR "${OPENCL_CFLAGS}")
separate_arguments(OPENCL_CFLAGS)


set(OPENCL_LIBS "${OPENCL_LIBS}" PARENT_SCOPE)
set(OPENCL_CFLAGS "${OPENCL_CFLAGS}" PARENT_SCOPE)
set(OPENCL_CFLAGS_STR "${OPENCL_CFLAGS_STR}" PARENT_SCOPE)

set(SANITIZER_LIBS "${SANITIZER_LIBS}" PARENT_SCOPE)
set(SANITIZER_OPTIONS "${SANITIZER_OPTIONS}" PARENT_SCOPE)
set(SANITIZER_OPTIONS_STR "${SANITIZER_OPTIONS_STR}" PARENT_SCOPE)

add_subdirectory("poclu")

