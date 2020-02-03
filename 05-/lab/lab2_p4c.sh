#!/bin/bash -e


if [[ -e ./src/lab2_vit ]] ; then
    binStr="./src/lab2_vit"
else
    echo "Couldn't find program to execute."
    exit 1
fi

echo "Decoding ..."
# Note p018k7.100.5.noloop.fsm is a hmm with transition prob. 
$binStr --gmm p4a.100.gmm --audio_file p018k7t.100.dat \
    --graph_file p018k7.100.5.noloop.fsm --word_syms p018k2.syms \
    --dcd_file p4c.100.dcd

echo "Computing WER ..."
#p018h1.calc-wer.sh p4c.100.dcd p018k7t.100.trn temp
./calc-wer.sh p018k7t.100.trn p4c.100.dcd > p4c.100.dcd.wer


