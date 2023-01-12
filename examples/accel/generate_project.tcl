# generate_project.tcl - Vivado script for generating Pynq-z1 bitstreams from tta cores
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





set mem_setup [lindex $argv 0]
set device_count [lindex $argv 1]
set use_axim [lindex $argv 2]
set project_postfix ${mem_setup}_${device_count}
set device0_offset 1073741824
set device_mmap_size 131072


#set script_path [ file dirname [ file normalize [ info script ] ] ]
set script_path [pwd]

set rtl_path "$script_path/rtls/rtl_$mem_setup"
set project_path "$script_path/vivado_$project_postfix"

create_project vivado_$project_postfix $project_path -part xc7z020clg400-1
set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
add_files [list $rtl_path/platform $rtl_path/gcu_ic $rtl_path/vhdl]

create_bd_design "toplevel"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {10.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.MMCM_DIVCLK_DIVIDE {2} CONFIG.MMCM_CLKFBOUT_MULT_F {15.625} CONFIG.MMCM_CLKOUT0_DIVIDE_F {78.125} CONFIG.RESET_PORT {resetn} CONFIG.CLKOUT1_JITTER {290.478} CONFIG.CLKOUT1_PHASE_ERROR {133.882}] [get_bd_cells clk_wiz_0]

if {${use_axim}} {
    set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]
    connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
}

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins clk_wiz_0/resetn]
connect_bd_net [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins clk_wiz_0/clk_out1]


for {set i 0} {$i < $device_count} {incr i} {
    create_bd_cell -type module -reference tta_core_toplevel tta_core_toplevel_${i}
    set_property -dict [list CONFIG.local_mem_addrw_g {13} ] [get_bd_cells tta_core_toplevel_${i}]
    if {${use_axim}} {
        set device_offset [expr ${device0_offset} + ${i} * ${device_mmap_size}]
        set_property -dict [list CONFIG.axi_offset_low_g ${device_offset}] [get_bd_cells tta_core_toplevel_${i}]
    }
    connect_bd_net [get_bd_pins tta_core_toplevel_${i}/clk] [get_bd_pins clk_wiz_0/clk_out1]
}



apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/clk_wiz_0/clk_out1 (10 MHz)} Clk_slave {/clk_wiz_0/clk_out1 (10 MHz)} Clk_xbar {/clk_wiz_0/clk_out1 (10 MHz)} Master {/processing_system7_0/M_AXI_GP0} Slave {/tta_core_toplevel_0/s_axi} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins tta_core_toplevel_0/s_axi]

if {${use_axim}} {
    set interface_count [expr ${device_count} + 1]
    set_property -dict [list CONFIG.NUM_MI ${interface_count} CONFIG.NUM_SI ${interface_count}] [get_bd_cells axi_smc]

    #connect device 0
    connect_bd_intf_net [get_bd_intf_pins axi_smc/S01_AXI] [get_bd_intf_pins tta_core_toplevel_0/m_axi]
    connect_bd_intf_net [get_bd_intf_pins axi_smc/M01_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
} else {
    set_property -dict [list CONFIG.NUM_MI ${device_count}] [get_bd_cells axi_smc]
}


connect_bd_net [get_bd_pins tta_core_toplevel_0/rstx] [get_bd_pins rst_clk_wiz_0_10M/peripheral_aresetn]

#connect other devices
for {set i 1} {$i < $device_count} {incr i} {
    set next [expr $i + 1]
    if {${use_axim}} {
        connect_bd_intf_net [get_bd_intf_pins axi_smc/S0${next}_AXI] [get_bd_intf_pins tta_core_toplevel_${i}/m_axi]
    }
    connect_bd_intf_net [get_bd_intf_pins axi_smc/M0${next}_AXI] [get_bd_intf_pins tta_core_toplevel_${i}/s_axi]
    connect_bd_net [get_bd_pins tta_core_toplevel_${i}/rstx] [get_bd_pins rst_clk_wiz_0_10M/peripheral_aresetn]
}

assign_bd_address

regenerate_bd_layout

make_wrapper -files [get_files $project_path/vivado_$project_postfix.srcs/sources_1/bd/toplevel/toplevel.bd] -top
add_files -norecurse $project_path/vivado_$project_postfix.gen/sources_1/bd/toplevel/hdl/toplevel_wrapper.v
set_property top toplevel_wrapper [current_fileset]
update_compile_order -fileset sources_1

save_bd_design
set_property strategy Flow_RuntimeOptimized [get_runs synth_1]
set_property strategy Flow_RuntimeOptimized [get_runs impl_1]

launch_runs impl_1 -to_step write_bitstream -jobs 8

wait_on_runs impl_1

open_run impl_1
if {[get_property SLACK [get_timing_paths]] >= 0} {
    puts "Design met timing."
    return
} else {
    error "Design failed to meet timing."
}

