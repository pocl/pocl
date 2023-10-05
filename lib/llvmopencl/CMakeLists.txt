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

# this is required, because we're compiling these files with LLVM-provided flags, which
# usually contain their own setting of C++ standard, which may be in conflict
# with pocl's default C++ standard.
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)

set(LLVMPASSES_SOURCES "AllocasToEntry.cc"
                       "AllocasToEntry.h"
                       "AutomaticLocals.cc"
                       "Barrier.h"
                       "BarrierTailReplication.cc"
                       "BarrierTailReplication.h"
                       "BreakConstantGEPs.cpp"
                       "BreakConstantGEPs.h"
                       "CanonicalizeBarriers.cc"
                       "CanonicalizeBarriers.h"
                       "DebugHelpers.cc"
                       "DebugHelpers.h"
                       "Flatten.cc"
                       "FlattenGlobals.cc"
                       "FlattenBarrierSubs.cc"
                       "HandleSamplerInitialization.cc"
                       "HandleSamplerInitialization.h"
                       "ImplicitConditionalBarriers.cc"
                       "ImplicitConditionalBarriers.h"
                       "ImplicitLoopBarriers.cc"
                       "ImplicitLoopBarriers.h"
                       "InlineKernels.cc"
                       "IsolateRegions.cc"
                       "IsolateRegions.h"
                       "Kernel.cc"
                       "Kernel.h"
                       "linker.cpp"
                       "linker.h"
                       "LLVMUtils.cc"
                       "LLVMUtils.h"
                       "LoopBarriers.cc"
                       "LoopBarriers.h"
                       "OptimizeWorkItemFuncCalls.cc"
                       "OptimizeWorkItemFuncCalls.h"
                       "PHIsToAllocas.cc"
                       "PHIsToAllocas.h"
                       "ParallelRegion.cc"
                       "ParallelRegion.h"
                       "ProgramScopeVariables.cc"
                       "ProgramScopeVariables.h"
                       "RemoveBarrierCalls.cc"
                       "RemoveBarrierCalls.h"
                       "RemoveOptnoneFromWIFunc.cc"
                       "RemoveOptnoneFromWIFunc.h"
                       "SubCFGFormation.cc"
                       "SubCFGFormation.h"
                       "UnifyPrintf.cc"
                       "UnifyPrintf.h"
                       "VariableUniformityAnalysis.cc"
                       "VariableUniformityAnalysis.h"
                       "WorkItemAliasAnalysis.cc"
                       "Workgroup.cc"
                       "Workgroup.h"
                       "WorkitemHandler.cc"
                       "WorkitemHandler.h"
                       "WorkitemHandlerChooser.cc"
                       "WorkitemHandlerChooser.h"
                       "WorkitemLoops.cc"
                       "WorkitemLoops.h"
                       "WorkitemReplication.cc"
                       "WorkitemReplication.h")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LLVM_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(../CL)
include_directories(../CL/devices)

if(MSVC)
  set_source_files_properties( ${LLVMPASSES_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)

# This library (just a bunch of objects really) is linked against libpocl
# to call the kernel compiler passes directly from the host library.
#noinst_LTLIBRARIES = libllvmpasses.la
add_library("llvmpasses" OBJECT ${LLVMPASSES_SOURCES})
harden("llvmpasses")

if(OCL_ICD_INCLUDE_DIRS)
  target_include_directories("llvmpasses" SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
endif()

option(ENABLE_LIBLLVMOPENCL "Build separate libllvmopencl.so for use with 'opt'" OFF)
if(ENABLE_LIBLLVMOPENCL)
# This creates a separate llvmopencl.so that is loaded to 'opt'
# for the pocl-workgroup version.
#pkglib_LTLIBRARIES = llvmopencl.la
add_library("llvmopencl" MODULE "$<TARGET_OBJECTS:llvmpasses>")
harden("llvmopencl")

# not sure where other platforms get their library linkage list, probably there is
# way to make this more consistent on different platforms
if(MSVC)
  target_link_libraries("llvmopencl" ${POCL_LLVM_LIBS})
endif(MSVC)

#target_link_libraries("llvmopencl" "${LLVM_SYSLIBS}")
endif(ENABLE_LIBLLVMOPENCL)
 
# TODO cmake should handle this, i might me missing something …
if(APPLE)
  set(ADD_LD_FLAGS "-Wl,-undefined -Wl,dynamic_lookup ")
else()
  set(ADD_LD_FLAGS "-Wl,-export-dynamic")
endif()

if(ENABLE_LIBLLVMOPENCL)
# TODO ENABLE_EXPORTS -> Wl,export_dynamic
# https://www.cmake.org/cmake/help/v2.8.12/cmake.html#prop_tgt:ENABLE_EXPORTS
# https://www.cmake.org/pipermail/cmake/2011-November/047708.html
set_target_properties("llvmopencl" PROPERTIES LINK_FLAGS "${LLVM_LDFLAGS} ${ADD_LD_FLAGS}")
if(NOT APPLE)
  set_target_properties("llvmopencl" PROPERTIES SOVERSION "${KERNEL_COMPILER_LIB_VERSION}" VERSION "${KERNEL_COMPILER_LIB_VERSION}")
endif()

install(TARGETS "llvmopencl"
        LIBRARY DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR}
        COMPONENT "lib"
        ARCHIVE DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR}/static
        COMPONENT "dev"
)
endif(ENABLE_LIBLLVMOPENCL)

if(ENABLE_IPO)
  set_property(TARGET "llvmpasses" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
if(ENABLE_LIBLLVMOPENCL)
  set_property(TARGET "llvmopencl" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif(ENABLE_LIBLLVMOPENCL)
endif()
