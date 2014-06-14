
include("bitcode_rules")

if(USE_VECMATHLIB)
  set(KERNEL_SOURCES ${SOURCES_WITH_VML})
else()
  set(KERNEL_SOURCES ${SOURCES_WITHOUT_VML})
endif()


# Use HOST flags:
#~ CLANG_FLAGS = @HOST_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off
separate_arguments(HOST_CLANG_FLAGS)
set(CLANG_FLAGS ${HOST_CLANG_FLAGS} "-Xclang" "-ffake-address-space-map" "-emit-llvm" "-ffp-contract=off")

# LLC_FLAGS   = @HOST_LLC_FLAGS@
separate_arguments(HOST_LLC_FLAGS)
set(LLC_FLAGS ${HOST_LLC_FLAGS})

# TODO LLC_flags is used by kernel.bc target, but ld is unused
#LD_FLAGS    = @HOST_LD_FLAGS@

# KERNEL_TARGET = @OCL_KERNEL_TARGET@
make_kernel_bc(KERNEL_BC "${OCL_KERNEL_TARGET}" ${KERNEL_SOURCES})

# just debug
message(STATUS "Host Kernel BC: ${KERNEL_BC}")

# a target is needed...
add_custom_target("kernel_host" ALL DEPENDS ${KERNEL_BC})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")


