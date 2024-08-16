#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
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

add_compile_options(${TCE_INCLUDES})
add_compile_options(${TCE_CXXFLAGS})

# If we compile with -pedantic, the #include_next in our
# CL headers result in a bunch of annoying warnings like:
#
#    warning: #include_next is a GCC extension
#
# ... which is impossible to disable in any other way
# than disabling -pedantic (AFAICT, with GCC 4/5)
add_compile_options("-Wno-pedantic")

add_subdirectory("printf")

add_subdirectory("autolocal")

# Produces now wrong results with LLVM 16 target-specific opts/autovec
# on. To port to the latest LLVM and investigate.
# add_subdirectory("fp16")
if(ENABLE_HOST_CPU_DEVICES)
  add_subdirectory("multi_AS_copy")
endif()

