#!/bin/bash -e


if [[ -e ./src/lab2_vit ]] ; then
    binStr="./src/lab2_vit"
else
    echo "Couldn't find program to execute."
    exit 1
fi


$binStr --gmm p018k7.22.20.gmm --audio_file p018k7t.10.dat \
    --graph_file p018k1.noloop.fsm --word_syms p018k2.syms \
    --dcd_file /dev/null


