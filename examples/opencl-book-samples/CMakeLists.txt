#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2015 pocl developers
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

set(TS_NAME "opencl-book-samples")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(HAVE_GIT)

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  #DOWNLOAD_COMMAND "/bin/true"
  GIT_REPOSITORY "https://github.com/bgaster/opencl-book-samples.git"
  PATCH_COMMAND pwd && echo Patching && sed -i "s/bool doCPU = false/bool doCPU = true/g" src/Chapter_16/Dijkstra/oclDijkstra.cpp &&
  sed -i "s/size_t localWorkSize = maxWorkGroupSize/size_t localWorkSize = 2/g" src/Chapter_16/Dijkstra/oclDijkstraKernel.cpp &&
  sed -i "s/device.j..type == CL_DEVICE_TYPE_GPU/device[j].type \\& CL_DEVICE_TYPE_GPU/g" src/Chapter_22/spmv.c &&
  sed -i "s/context.CL_DEVICE_TYPE_GPU/context\(CL_DEVICE_TYPE_CPU/g" src/Chapter_12/VectorAdd/vecadd.cpp &&
  rm -f src/Chapter_12/Sinewave/CMakeLists.txt

  # update restores src/Chapter_12/Sinewave/CMakeLists.txt, so disable it
  UPDATE_COMMAND "/bin/true"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=RelWithDebInfo
  "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=-g -O3 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
  "-DCMAKE_EXE_LINKER_FLAGS=-lGL -lglut -pthread"

  INSTALL_COMMAND "/bin/true"
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})


add_test(NAME "opencl_book_samples_HelloWorld"
         COMMAND "${TS_BUILDDIR}/src/Chapter_2/HelloWorld/HelloWorld"
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_2/HelloWorld")
add_test(NAME "opencl_book_samples_OpenCLInfo"
         COMMAND "${TS_BUILDDIR}/src/Chapter_3/OpenCLInfo/OpenCLInfo"
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_3/OpenCLInfo")
add_test(NAME "opencl_book_samples_OpenCLConvolutionChap3"
         COMMAND "${TS_BUILDDIR}/src/Chapter_3/OpenCLConvolution/OpenCLConvolutionChap3"
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_3/OpenCLConvolution")
add_test(NAME "opencl_book_samples_HelloBinaryWorld"
         COMMAND "${TS_BUILDDIR}/src/Chapter_6/HelloBinaryWorld/HelloBinaryWorld"
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_6/HelloBinaryWorld")
add_test(NAME "opencl_book_samples_SimpleBufferSubBuffer"
         COMMAND "${TS_BUILDDIR}/src/Chapter_7/SimpleBufferSubBuffer/SimpleBufferSubBuffer"
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_7/SimpleBufferSubBuffer")
# requires input images
#add_test(NAME "opencl_book_samples_ImageFilter2D"
#         COMMAND "${TS_BUILDDIR}/src/Chapter_8/ImageFilter2D/ImageFilter2D"
#         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_8/ImageFilter2D")
add_test(NAME "opencl_book_samples_vecadd"
         COMMAND "${TS_BUILDDIR}/src/Chapter_12/VectorAdd/vecadd"
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_12/VectorAdd")
# requires a GPU
#add_test(NAME "opencl_book_samples_histogram"
#         COMMAND "${TS_BUILDDIR}/src/Chapter_14/histogram/histogram"
#         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_14/histogram")
add_test(NAME "opencl_book_samples_Dijkstra"
         COMMAND "${TS_BUILDDIR}/src/Chapter_16/Dijkstra/Dijkstra" --cpu
         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_16/Dijkstra")
# doesn't work
#add_test(NAME "opencl_book_samples_spmv"
#         COMMAND "${TS_BUILDDIR}/src/Chapter_22/spmv"
#         WORKING_DIRECTORY "${TS_SRCDIR}/src/Chapter_22")




set_tests_properties(
  opencl_book_samples_HelloWorld
  opencl_book_samples_OpenCLInfo
  opencl_book_samples_OpenCLConvolutionChap3
  opencl_book_samples_HelloBinaryWorld
  opencl_book_samples_SimpleBufferSubBuffer
#  opencl_book_samples_ImageFilter2D
  opencl_book_samples_vecadd
#  opencl_book_samples_histogram
  opencl_book_samples_Dijkstra
#  opencl_book_samples_spmv
  PROPERTIES
    LABELS "opencl-book-samples")


else()
  message(STATUS "Disabling opencl-book-samples, requires Subversion to checkout the sources")
endif()
