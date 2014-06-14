
include("bitcode_rules")

# Use the libc (newlib) printf() for now because the pocl's
# fails in TCE code gen:
# SplitVectorResult #0: 0x24c5ae0: v8i16,ch = vaarg 0x20628e8, 0x2423ed0, 0x24c59e0, 0x24c4fe0 [ORD=223] [ID=0]
# LLVM ERROR: Do not know how to split the result of this operator!
if(NEW_PRINTF_WORKS)
  list(REMOVE_ITEM SOURCES_WITHOUT_VML "printf.c")
  list(APPEND SOURCES_WITHOUT_VML "printf_constant.c")
endif()



# Use TARGET flags:
# with some additional CLANG_FLAGS
#CLANG_FLAGS = @TARGET_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off
separate_arguments(TCE_TARGET_CLANG_FLAGS)
set(CLANG_FLAGS ${TCE_TARGET_CLANG_FLAGS} "-Xclang" "-ffake-address-space-map" "-emit-llvm" "-ffp-contract=off" "-target" "tce-tut-llvm" "-isystem" "${TCE_PREFIX}/tce-llvm/include")

#LLC_FLAGS   = @TARGET_LLC_FLAGS@
separate_arguments(TCE_TARGET_LLC_FLAGS)
set(LLC_FLAGS ${TCE_TARGET_LLC_FLAGS})

# TODO LLC_flags is used by kernel.bc target, but ld is unused
#LD_FLAGS    = @TARGET_LD_FLAGS@

#KERNEL_TARGET = tce (WRONG)
make_kernel_bc(KERNEL_BC "tce-tut-llvm" ${SOURCES_WITHOUT_VML})

# just debug
message(STATUS "TCE Kernel BC: ${KERNEL_BC}")

# a target is needed...
add_custom_target("kernel_tce" ALL DEPENDS ${KERNEL_BC})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")




