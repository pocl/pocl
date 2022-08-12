
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bitstreams)

add_custom_command(OUTPUT vitis_vecadd_0
    COMMAND rm -rf vitis_vecadd_0
    COMMAND /home/topi/Xilinx/Vitis_HLS/2021.2/bin/vitis_hls -f ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_core.tcl -tclargs ${CMAKE_CURRENT_SOURCE_DIR} -tclargs 1
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/poclAccel.cpp ${CMAKE_CURRENT_SOURCE_DIR}/poclAccel.h ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_core.tcl)
add_custom_target(vitis_vecadd DEPENDS vitis_vecadd_0)

add_custom_command(OUTPUT bitstreams/vecadd_1.bit
    COMMAND rm -rf vivado_vecadd_1
    COMMAND /home/topi/Xilinx/Vivado/2021.2/bin/vivado -mode batch -source ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_project.tcl -tclargs vecadd -tclargs 1
    COMMAND cp vivado_vecadd_1/vivado_vecadd_1.runs/impl_1/toplevel_wrapper.bit bitstreams/vecadd_1.bit
    COMMAND cp vivado_vecadd_1/vivado_vecadd_1.gen/sources_1/bd/toplevel/hw_handoff/toplevel.hwh bitstreams/vecadd_1.hwh
    DEPENDS vitis_vecadd ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_project.tcl)
add_custom_target(hls_vecadd_bs DEPENDS bitstreams/vecadd_1.bit)

add_dependencies(bitstreams hls_vecadd_bs)
