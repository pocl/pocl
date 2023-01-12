#!/bin/bash

if [ -z "$1" ]
then
    echo "Machine name is not set. Please set it as the first argument to the script"
    exit 1
fi

if [ "${1:0:4}" = axim ]
then
    DMEM_OFFSET=0x40018000
else
    DMEM_OFFSET=0
fi

STANDALONE_GLOBAL_AS_OFFSET=${DMEM_OFFSET} bash standalone_0_build
generatebits --data-start=param,${DMEM_OFFSET} -d -w 4 -e tta_core -p standalone.tpef ../../../tools/data/tta_test_machines/${1}.adf
COMPL_SIGNAL_ADDR=$(echo "mach ../../../tools/data/tta_test_machines/${1}.adf;prog standalone.tpef;symbol_address __completion_signal" |\
    ttasim | head -n1)

COMPL_SIGNAL_ADDR=$((COMPL_SIGNAL_ADDR - DMEM_OFFSET))

pushd rtls/rtl_${1} > /dev/null
bash ghdl_compile.sh -v93c -a
bash ghdl_platform_compile.sh -v93c
bash ghdl_simulate.sh -v93c -r 2000000 -n tta_almaif_tb -g "imem_image=../../standalone.img"\
    -g "dmem_image=../../standalone_param.img" -g "completion_signal_address_g=${COMPL_SIGNAL_ADDR}"\
    -g "dmem_dump_path=../../rtlsim_raw_output.txt"
popd > /dev/null

OUTPUT_BUFFER_SYMBOL=$(tail -n1 standalone_0_ttasim |cut -d' ' -f6 |cut -d';' -f1)

OUTPUT_BUFFER_ADDR=$(echo "mach ../../../tools/data/tta_test_machines/${1}.adf;prog standalone.tpef;symbol_address ${OUTPUT_BUFFER_SYMBOL}" |\
    ttasim | head -n1)


OUTPUT_BUFFER_ADDR=$((OUTPUT_BUFFER_ADDR - DMEM_OFFSET))

OUTPUT_BUFFER_IDX=$((OUTPUT_BUFFER_ADDR/4))


cat "rtlsim_raw_output.txt" | head -n$((OUTPUT_BUFFER_IDX+128)) |tail -n128
rm  -f "rtlsim_raw_output.txt"
