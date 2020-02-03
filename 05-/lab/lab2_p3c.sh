#!/bin/bash -e


if [[ -e ./src/lab2_fb ]] ; then
    binStr="./src/lab2_fb"
else
    echo "Couldn't find program to execute."
    exit 1
fi


$binStr --audio_file p018k7.22.dat --graph_file p018k7.22.fsm --iters 20 \
    --in_gmm p018k1.gmm.dat --out_gmm p3c.gmm


