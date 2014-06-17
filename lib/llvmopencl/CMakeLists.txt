set(LLVMPASSES_SOURCES "Barrier.h"
                        "BarrierBlock.h" "BarrierBlock.cc"
                        "Kernel.h" "Kernel.cc"
                        "ParallelRegion.h" "ParallelRegion.cc"
                        "CanonicalizeBarriers.h" "CanonicalizeBarriers.cc"
                        "LoopBarriers.h" "LoopBarriers.cc"
                        "GenerateHeader.cc" "Workgroup.h" "Workgroup.cc"
            "BarrierTailReplication.h" "BarrierTailReplication.cc"
                        "Flatten.cc" "IsolateRegions.h" "IsolateRegions.cc"
            "WorkitemReplication.h" "WorkitemReplication.cc"
            "ImplicitLoopBarriers.h" "ImplicitLoopBarriers.cc"
            "WorkItemAliasAnalysis.cc" "WIVectorize.cc"
            "WorkitemHandler.h" "WorkitemHandler.cc"
            "WorkitemLoops.h" "WorkitemLoops.cc"
            "PHIsToAllocas.h" "PHIsToAllocas.cc"
            "BreakConstantGEPs.h" "BreakConstantGEPs.cpp"
            "WorkitemHandlerChooser.h" "WorkitemHandlerChooser.cc"
            "AllocasToEntry.h" "AllocasToEntry.cc"
            "TargetAddressSpaces.h" "TargetAddressSpaces.cc"
            "LLVMUtils.cc" "LLVMUtils.h"
            "VariableUniformityAnalysis.h" "VariableUniformityAnalysis.cc"
            "AutomaticLocals.cc" "ImplicitConditionalBarriers.cc"
            "ImplicitConditionalBarriers.h"
            "DebugHelpers.h" "DebugHelpers.cc")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LLVM_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")

# This library (just a bunch of objects really) is linked against libpocl
# to call the kernel compiler passes directly from the host library.
#noinst_LTLIBRARIES = libllvmpasses.la
add_library("llvmpasses" OBJECT ${LLVMPASSES_SOURCES})

# This creates a separate llvmopencl.so that is loaded to 'opt'
# for the pocl-workgroup version.
#pkglib_LTLIBRARIES = llvmopencl.la
add_library("llvmopencl" MODULE "$<TARGET_OBJECTS:llvmpasses>")

# TODO cmake should handle this, i might me missing something …
if(APPLE)
  set(ADD_LD_FLAGS "-Wl,-undefined -Wl,dynamic_lookup ")
else()
  set(ADD_LD_FLAGS "-Wl,-export-dynamic")
endif()

# TODO ENABLE_EXPORTS -> Wl,export_dynamic
# http://www.cmake.org/cmake/help/v2.8.12/cmake.html#prop_tgt:ENABLE_EXPORTS
# http://www.cmake.org/pipermail/cmake/2011-November/047708.html
set_target_properties("llvmopencl" PROPERTIES LINK_FLAGS "${LLVM_LDFLAGS} ${ADD_LD_FLAGS}")
if(NOT APPLE)
  set_target_properties("llvmpasses" "llvmopencl" PROPERTIES SOVERSION "${KERNEL_COMPILER_LIB_VERSION}" VERSION "$KERNEL_COMPILER_LIB_VERSION}" POSITION_INDEPENDENT_CODE ON)
endif()

install(TARGETS "llvmopencl"
        LIBRARY DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR}
        ARCHIVE DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR}/static)
