# generate_hls_core.tcl - Vitis HLS script for converting poclAccel.cpp -> RTL accelerator
#
#   Copyright (c) 2022 Topi Leppänen / Tampere University
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#   IN THE SOFTWARE.



set input_folder [lindex $argv 2]
set device_count [lindex $argv 3]

#0x40000000
set base_address 1073741824 
#0x8000
set dev_offset 32768


puts $device_count
for {set i 0} {$i < $device_count} {incr i} {

    set dev_address [expr ${base_address} + ${i} * ${dev_offset}] 

    open_project vitis_vecadd_${i}
    set_top poclAccel
    add_files ${input_folder}/poclAccel.cpp -cflags "-DBASE_ADDRESS=${dev_address} -DMEM_MAX_SIZE_BYTES=${dev_offset}"
    open_solution "solution1" -flow_target vivado
    set_part {xc7z020-clg400-1}
    create_clock -period 10 -name default
    config_export -format ip_catalog -output vecadd_hls_${i} -rtl verilog
    config_export -ipname poclAccel${i}
    config_interface -m_axi_addr64=0 -m_axi_conservative_mode

    set_directive_top -name poclAccel "poclAccel"
    #source "./mic_in/solution1/directives.tcl"
    #csim_design
    csynth_design
    #cosim_design
    export_design -rtl verilog -format ip_catalog -output vecadd_hls_${i}
}

exit
