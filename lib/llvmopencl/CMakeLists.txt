#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2024 pocl developers
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
                       "CanonicalizeBarriers.cc"
                       "CanonicalizeBarriers.h"
                       "DebugHelpers.cc"
                       "DebugHelpers.h"
                       "EmitPrintf.cc"
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
                       "MinLegalVecSize.cc"
                       "OptimizeWorkItemGVars.cc"
                       "OptimizeWorkItemGVars.h"
                       "PHIsToAllocas.cc"
                       "PHIsToAllocas.h"
                       "ParallelRegion.cc"
                       "ParallelRegion.h"
                       "ProgramScopeVariables.cc"
                       "ProgramScopeVariables.h"
                       "RemoveBarrierCalls.cc"
                       "RemoveBarrierCalls.h"
                       "SanitizeUBofDivRem.cc"
                       "SanitizeUBofDivRem.h"
                       "SubCFGFormation.cc"
                       "SubCFGFormation.h"
                       "UnreachablesToReturns.cc"
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
                       "WorkitemLoops.h")

if (ENABLE_REMOTE_SERVER)
  list(APPEND LLVMPASSES_SOURCES "SVMOffset.cc")
endif()

# add library of llvm passes (the object form is then linked to libpocl)
pocl_build_clang_llvm_object(llvmpasses ${LLVMPASSES_SOURCES})

target_include_directories(llvmpasses PRIVATE "${CMAKE_SOURCE_DIR}/lib/CL")
target_include_directories(llvmpasses PRIVATE "${CMAKE_SOURCE_DIR}/lib/CL/devices")

if(OCL_ICD_INCLUDE_DIRS)
  target_include_directories("llvmpasses" SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
endif()
