# generate_hls_project.tcl - Vivado script for generating Pynq-z1 bitstreams from hls cores
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
set project_postfix ${mem_setup}_${device_count}



#set script_path [ file dirname [ file normalize [ info script ] ] ]
set script_path [pwd]


set project_path "$script_path/vivado_$project_postfix"

create_project vivado_$project_postfix $project_path -part xc7z020clg400-1
set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]

create_bd_design "toplevel"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0


set_property -dict [list CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {10.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.MMCM_DIVCLK_DIVIDE {2} CONFIG.MMCM_CLKFBOUT_MULT_F {15.625} CONFIG.MMCM_CLKOUT0_DIVIDE_F {78.125} CONFIG.RESET_PORT {resetn} CONFIG.CLKOUT1_JITTER {290.478} CONFIG.CLKOUT1_PHASE_ERROR {133.882}] [get_bd_cells clk_wiz_0]


apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]


set ip_path ""
for {set i 0} {$i < $device_count} {incr i} {
    append ip_path "$script_path/vitis_vecadd_${i} "
}

set_property  ip_repo_paths  $ip_path [current_project]
update_ip_catalog


create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 util_vector_logic_0
set_property -dict [list CONFIG.C_SIZE {1} CONFIG.C_OPERATION {not} CONFIG.LOGO_FILE {data/sym_notgate.png}] [get_bd_cells util_vector_logic_0]




for {set i 0} {$i < $device_count} {incr i} {
    create_bd_cell -type ip -vlnv xilinx.com:hls:poclAccel${i}:1.0 poclAccel_${i}
    create_bd_cell -type ip -vlnv xilinx.com:ip:c_counter_binary:12.0 c_counter_binary_${i}

    set_property -dict [list CONFIG.SCLR {true}] [get_bd_cells c_counter_binary_${i}]
    connect_bd_net [get_bd_pins util_vector_logic_0/Res] [get_bd_pins c_counter_binary_${i}/SCLR]
    set_property -dict [list CONFIG.Output_Width {64}] [get_bd_cells c_counter_binary_${i}]
    connect_bd_net [get_bd_pins c_counter_binary_${i}/Q] [get_bd_pins poclAccel_${i}/cycle_counter]

    create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_${i}
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_${i}

    set_property -dict [list CONFIG.Memory_Type {True_Dual_Port_RAM} CONFIG.Enable_B {Use_ENB_Pin} CONFIG.Use_RSTB_Pin {true} CONFIG.Port_B_Clock {100} CONFIG.Port_B_Write_Rate {50} CONFIG.Port_B_Enable_Rate {100}] [get_bd_cells blk_mem_gen_${i}]
    set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1}] [get_bd_cells axi_bram_ctrl_${i}]


    connect_bd_net [get_bd_pins poclAccel_${i}/ap_clk] [get_bd_pins clk_wiz_0/clk_out1]


    connect_bd_net [get_bd_pins axi_bram_ctrl_${i}/s_axi_aclk] [get_bd_pins clk_wiz_0/clk_out1]

    connect_bd_net [get_bd_pins c_counter_binary_${i}/CLK] [get_bd_pins clk_wiz_0/clk_out1]

    connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_${i}/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_${i}/BRAM_PORTA]
    connect_bd_intf_net [get_bd_intf_pins blk_mem_gen_${i}/BRAM_PORTB] [get_bd_intf_pins poclAccel_${i}/Control_PORTA]
}


connect_bd_net [get_bd_pins clk_wiz_0/clk_in1] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins clk_wiz_0/clk_out1]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/clk_wiz_0/clk_out1 (10 MHz)} Clk_slave {/clk_wiz_0/clk_out1 (10 MHz)} Clk_xbar {/clk_wiz_0/clk_out1 (10 MHz)} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_bram_ctrl_0/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]
connect_bd_net [get_bd_pins util_vector_logic_0/Op1] [get_bd_pins rst_clk_wiz_0_10M/peripheral_aresetn]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_clk_wiz_0_10M/ext_reset_in]

set if_count [expr $device_count + 1]
set_property -dict [list CONFIG.NUM_MI ${if_count} CONFIG.NUM_SI ${if_count}] [get_bd_cells axi_smc]

connect_bd_intf_net [get_bd_intf_pins poclAccel_0/m_axi_output_r] [get_bd_intf_pins axi_smc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M01_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_net [get_bd_pins poclAccel_0/ap_rst_n] [get_bd_pins rst_clk_wiz_0_10M/peripheral_aresetn]


#TODO TEST THIS, not used for now
for {set i 1} {$i < $device_count} {incr i} {
    set next [expr $i + 1]
    connect_bd_intf_net [get_bd_intf_pins poclAccel_${i}/m_axi_output_r] [get_bd_intf_pins axi_smc/S0${next}_AXI]
    connect_bd_intf_net [get_bd_intf_pins axi_smc/M0${next}_AXI] [get_bd_intf_pins axi_bram_ctrl_${i}/S_AXI]

    connect_bd_net [get_bd_pins poclAccel_${i}/ap_rst_n] [get_bd_pins rst_clk_wiz_0_10M/peripheral_aresetn]
    connect_bd_net [get_bd_pins axi_bram_ctrl_${i}/s_axi_aresetn] [get_bd_pins rst_clk_wiz_0_10M/peripheral_aresetn]
}


set base_address 1073741824
set dev_offset 32768

for {set i 0} {$i < $device_count} {incr i} {
    set dev_address [expr ${base_address} + ${i} * ${dev_offset}]
    assign_bd_address -target_address_space /processing_system7_0/Data [get_bd_addr_segs axi_bram_ctrl_${i}/S_AXI/Mem0] -force
    set_property offset $dev_address [get_bd_addr_segs processing_system7_0/Data/SEG_axi_bram_ctrl_${i}_Mem0]
    set_property range 32K [get_bd_addr_segs processing_system7_0/Data/SEG_axi_bram_ctrl_${i}_Mem0]
}

assign_bd_address

set last_dev [expr $device_count - 1]
set last_address [expr $base_address + ${last_dev} * ${dev_offset}]

for {set i 0} {$i < $device_count} {incr i} {
    puts $i
    set_property offset $last_address [get_bd_addr_segs poclAccel_${i}/Data_m_axi_output_r/SEG_axi_bram_ctrl_${last_dev}_Mem0]
    set_property range 32K [get_bd_addr_segs poclAccel_${i}/Data_m_axi_output_r/SEG_axi_bram_ctrl_${last_dev}_Mem0]
}



apply_bd_automation -rule xilinx.com:bd_rule:board -config { Manual_Source {Auto}}  [get_bd_pins clk_wiz_0/resetn]
regenerate_bd_layout


make_wrapper -files [get_files $project_path/vivado_$project_postfix.srcs/sources_1/bd/toplevel/toplevel.bd] -top
add_files -norecurse $project_path/vivado_$project_postfix.gen/sources_1/bd/toplevel/hdl/toplevel_wrapper.v
set_property top toplevel_wrapper [current_fileset]
update_compile_order -fileset sources_1

save_bd_design
set_property strategy Flow_RuntimeOptimized [get_runs synth_1]
set_property strategy Flow_RuntimeOptimized [get_runs impl_1]

launch_runs impl_1 -to_step write_bitstream -jobs 8

wait_on_run impl_1

open_run impl_1
if {[get_property SLACK [get_timing_paths]] >= 0} {
    puts "Design met timing."
    return
} else {
    error "Design failed to meet timing."
}

