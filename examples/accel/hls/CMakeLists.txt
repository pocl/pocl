#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2022 Topi Leppänen
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

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bitstreams)

if (DEFINED VITIS_HLS_PATH)
add_custom_command(OUTPUT vitis_vecadd_0
    COMMAND rm -rf vitis_vecadd_0
    COMMAND ${VITIS_HLS} -f ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_core.tcl -tclargs ${CMAKE_CURRENT_SOURCE_DIR} -tclargs 1
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/poclAccel.cpp ${CMAKE_CURRENT_SOURCE_DIR}/poclAccel.h ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_core.tcl)
add_custom_target(vitis_vecadd DEPENDS vitis_vecadd_0)

add_custom_command(OUTPUT bitstreams/vecadd_1.bit
    COMMAND rm -rf vivado_vecadd_1
    COMMAND ${VIVADO} -mode batch -source ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_project.tcl -tclargs vecadd -tclargs 1
    COMMAND cp vivado_vecadd_1/vivado_vecadd_1.runs/impl_1/toplevel_wrapper.bit bitstreams/vecadd_1.bit
    COMMAND cp vivado_vecadd_1/vivado_vecadd_1.gen/sources_1/bd/toplevel/hw_handoff/toplevel.hwh bitstreams/vecadd_1.hwh
    DEPENDS vitis_vecadd ${CMAKE_CURRENT_SOURCE_DIR}/generate_hls_project.tcl)
add_custom_target(hls_vecadd_bs DEPENDS bitstreams/vecadd_1.bit)

add_dependencies(bitstreams hls_vecadd_bs)

endif ()

if (PYNQ_AVAILABLE)
	add_test(NAME "upload_bitstream/pynq_hls" COMMAND sudo -E python -c "from pynq import Overlay;Overlay(\'${CMAKE_CURRENT_BINARY_DIR}/bitstreams/vecadd_1.bit\')")

    set_tests_properties( "upload_bitstream/pynq_hls"
        PROPERTIES
        RESOURCE_LOCK fpga
        FIXTURES_SETUP upload_hls_bitstream
    )

    add_test(NAME "examples/accel/add.i32/pynq_hls" COMMAND sudo -E ${CMAKE_CURRENT_BINARY_DIR}/../accel_example pocl.add.i32)
    add_test(NAME "examples/accel/mul.i32/pynq_hls" COMMAND sudo -E ${CMAKE_CURRENT_BINARY_DIR}/../accel_example pocl.mul.i32)

    set_tests_properties( "examples/accel/add.i32/pynq_hls" "examples/accel/mul.i32/pynq_hls"
            PROPERTIES
                PASS_REGULAR_EXPRESSION "OK"
                LABELS "custom_device;almaif"
                ENVIRONMENT "POCL_DEVICES=almaif;POCL_ALMAIF0_PARAMETERS=0x40000000,dummy,1,2;POCL_ALMAIF0_HASH=${tta_device_hash}"
                DEPENDS "pocl_version_check;examples/example0"
                RESOURCE_LOCK fpga
                FIXTURES_REQUIRED upload_hls_bitstream
    )
endif ()
