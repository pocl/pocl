

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
