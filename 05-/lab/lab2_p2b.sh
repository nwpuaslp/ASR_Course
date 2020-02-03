#!/bin/bash -e


if [[ -e ./src/lab2_train ]] ; then
    binStr="./src/lab2_train"
else
    echo "Couldn't find program to execute."
    exit 1
fi


$binStr --audio_file p018k7.22.dat --align_file p018k7.22.2.align \
    --iters 1 --in_gmm p018k1.gmm.dat --out_gmm p2b.gmm


