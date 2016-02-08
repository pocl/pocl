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

add_subdirectory("kernel")

add_subdirectory("llvmopencl")

add_subdirectory("CL")

#############################################################

# must be after tce b/c of LD_FLAGS_BIN
# and after adding lib/CL subdir b/c of libpocl location,
# but before poclu & examples & tests (they need these variables)

set(OPENCL_CFLAGS "${PTHREAD_CFLAGS}")

if(TESTS_USE_ICD)

  if(OPENCL_FOUND)
    set(OPENCL_LIBS ${OPENCL_LIBRARIES})
  else()
    message(WARNING "Tests-use-ICD requested, but cannot find an OpenCL library; adding -lOpenCL to LDFLAGS")
    set(OPENCL_LIBS "OpenCL")
  endif()

  set(OPENCL_LIBS "${PTHREAD_LDFLAGS};${OPENCL_LIBS}")

  set(OPENCL_EXTLIBS ${OPENCL_LIBS})
  set(OPENCL_CMAKE_OPTIONS "")

else()

  # used by examples / tests

  #OPENCL_LIBS='$(abs_top_builddir)/lib/CL/libpocl.la $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
  set(OPENCL_LIBS ${LD_FLAGS_BIN} "OpenCL")

  # TODO used by Rodinia makefile.am ?
  #OPENCL_EXTLIBS='-L$(abs_top_builddir)/lib/CL/.libs -lpocl -Xlinker -rpath=${abs_top_builddir}/lib/CL/.libs $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
  set(OPENCL_EXTLIBS "TODO_FIX_THIS")

  #~ OPENCL_CMAKE=' \
  #~ -D OPENCL_LIBRARIES:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libpocl.so \
  #~ -D OPENCL_LIBRARY:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libpocl.so \
  #~ -D CMAKE_EXE_LINKER_FLAGS:STRING='${CMAKE_THREAD_LIBS_INIT}' \
  #~ -D OPENCL_INCLUDE_DIRS:PATH=${abs_top_srcdir}/include \
  #~ -D OPENCL_INCLUDE_DIR:PATH=${abs_top_srcdir}/include
  #~ '
  set(OPENCL_CMAKE_OPTIONS "-D OPENCL_LIBRARIES:FILEPATH='$<TARGET_FILE:pocl>'
                            -D OPENCL_LIBRARY:FILEPATH='$<TARGET_FILE:pocl>'
                            -D CMAKE_EXE_LINKER_FLAGS:STRING='${CMAKE_THREAD_LIBS_INIT}'
                            -D OPENCL_INCLUDE_DIRS:PATH='${CMAKE_SOURCE_DIR}/include'
                            -D OPENCL_INCLUDE_DIR:PATH='${CMAKE_SOURCE_DIR}/include' ")

endif()

set(OPENCL_LIBS "${OPENCL_LIBS}" PARENT_SCOPE)
set(OPENCL_EXTLIBS "${OPENCL_EXTLIBS}" PARENT_SCOPE)
set(OPENCL_CMAKE_OPTIONS "${OPENCL_CMAKE_OPTIONS}" PARENT_SCOPE)
set(OPENCL_CFLAGS "${OPENCL_CFLAGS}" PARENT_SCOPE)

add_subdirectory("poclu")

