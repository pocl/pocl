
include("bitcode_rules")

# Use TARGET flags:
#CLANG_FLAGS = @TARGET_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off
separate_arguments(CELL_TARGET_CLANG_FLAGS)
set(CLANG_FLAGS ${CELL_TARGET_CLANG_FLAGS} "-Xclang" "-ffake-address-space-map" "-emit-llvm" "-ffp-contract=off")

#LLC_FLAGS   = @TARGET_LLC_FLAGS@
separate_arguments(CELL_TARGET_LLC_FLAGS)
set(LLC_FLAGS ${CELL_TARGET_LLC_FLAGS})

# TODO LLC_flags is used by kernel.bc target, but ld is unused
#LD_FLAGS    = @TARGET_LD_FLAGS@

#KERNEL_TARGET = tce (WRONG)
make_kernel_bc(KERNEL_BC "cellspu" ${SOURCES_WITHOUT_VML})

# just debug
message(STATUS "Cell SPU Kernel BC: ${KERNEL_BC}")

# a target is needed...
add_custom_target("kernel_cell" ALL DEPENDS ${KERNEL_BC})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")
