#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2021 pocl developers
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

if(MSVC)
  set_source_files_properties( pocl_ventus.h pocl_ventus.c ventus.h PROPERTIES LANGUAGE CXX )
endif(MSVC)
#message(STATUS "now in " "${CMAKE_CURRENT_SOURCE_DIR}")
#set(POCL_PUBLIC_LINK_LIST $ENV{VENTUS_INSTALL_PREFIX}/lib/ PARENT_SCOPE)
#
#list(APPEND POCL_LIBRARY_NAME "spike_driver" PARENT_SCOPE)
#set(POCL_LIBRARY_NAME "spike_driver:${POCL_LIBRARY_NAME}" PARENT_SCOPE)
#message(STATUS "devices library linked a library name: " ${POCL_PUBLIC_LINK_LIST} ${POCL_PRIVATE_LINK_LIST})
#message(STATUS "libraries need to link when generating pocl device library " ${name} ":" ${POCL_LIBRARY_NAME})

add_pocl_device_library("pocl-devices-ventus" pocl_ventus.h pocl_ventus.c ventus.h)
target_link_libraries(pocl-devices-ventus PRIVATE spike_driver)
target_link_directories(pocl-devices-ventus PRIVATE $ENV{VENTUS_INSTALL_PREFIX}/lib)