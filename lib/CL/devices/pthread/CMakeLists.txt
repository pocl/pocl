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

# To add a new BiK to pthread, add it to 4 places (needs to be refactored to
# reside in 1 place only):
# 2x common_driver.c (pocl_driver_free_program() and pocl_driver_add_host_builtins())
# 1x pthread_scheduler.c (pocl_pthread_prepare_kernel())
# 1x pthread.c (pocl_pthread_init())

# TODO: The options below should be probably made device-specific (they exist in
# basic device as well)

# these are needed in order to have POCL messages working in the BiKs
set(LOGGING_SOURCES ../../pocl_debug.h ../../pocl_debug.c ../../pocl_threads.c ../../pocl_threads.h ../../pocl_timing.c ../../pocl_timing.h)

if(ANDROID)
    find_library(ANDROID_LOG_LIB log REQUIRED)
else(ANDROID)
    set(ANDROID_LOG_LIB "")
endif(ANDROID)

function(add_pocl_host_builtin_library name)
    add_library(${name} SHARED ${ARGN})
    # make the logging files private
    target_sources(${name} PRIVATE ${LOGGING_SOURCES})
    harden("${name}")
    set_target_properties(${name} PROPERTIES PREFIX "lib" SUFFIX ".so")
    target_link_libraries(${name} PUBLIC ${ANDROID_LOG_LIB})
    install(TARGETS ${name} LIBRARY DESTINATION "${POCL_INSTALL_PRIVATE_LIBDIR}" COMPONENT "lib")
endfunction()

### Vector addition test BiK
set(VECADD_BIK_LIB pocl_pthread_add_i8)
add_pocl_host_builtin_library(${VECADD_BIK_LIB}
        builtin-kernels/pocl_add_i8.cpp
        builtin-kernels/pocl_add_i8.h)
set_target_properties(${VECADD_BIK_LIB} PROPERTIES LINKER_LANGUAGE CXX)

### Other pthread stuff
if (MSVC)
    set_source_files_properties(pocl-pthread.h pthread.c pthread_scheduler.c PROPERTIES LANGUAGE CXX)
endif (MSVC)

set(SOURCES pocl-pthread.h pthread.c pthread_scheduler.c builtin-kernels/metadata.h)
if (APPLE)
    list(APPEND SOURCES pthread_barrier.c)
endif (APPLE)
add_pocl_device_library(pocl-devices-pthread ${SOURCES})

if (ENABLE_LOADABLE_DRIVERS)
    target_link_libraries(pocl-devices-pthread PRIVATE pocl-devices-basic ${PTHREAD_LIBRARY})
endif ()

if(ENABLE_HOST_CPU_DEVICES_OPENMP)
    target_link_libraries(pocl-devices-pthread PRIVATE OpenMP::OpenMP_C)
endif()
