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
ttasim <standalone_0_ttasim |\
        tr ' ' '\n' |\
        tail -n129 |\
        head -n128|\
        xargs -i printf "%d\n" {}
