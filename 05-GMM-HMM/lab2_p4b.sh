#!/bin/bash -e


echo "======================================================================"
echo "Existing model trained on 300 sentences, isolated digit strings."
echo "======================================================================"

if [[ -e ./src/lab2_vit ]] ; then
    binStr="./src/lab2_vit"
else
    echo "Couldn't find program to execute."
    exit 1
fi

echo "Decoding ..."
$binStr --gmm p4a.300.gmm --audio_file p018k7tc.100.dat \
    --graph_file p018k1.loop.fsm --word_syms p018k2.syms \
    --dcd_file p4b.300.dcd

echo "Computing WER ..."
./calc-wer.sh p018k7tc.100.trn p4b.300.dcd > p4b.300.dcd.wer


echo "======================================================================"
echo "Training: 300 sentences, continuous digit strings."
echo "======================================================================"

if [[ -e ./src/lab2_fb ]] ; then
    binStr="./src/lab2_fb"
else
    echo "Couldn't find program to execute."
    exit 1
fi

$binStr --audio_file p018k7c.300.dat --graph_file p018k7c.300.fsm --iters 5 \
    --in_gmm p018k1.gmm.dat --out_gmm p4b.300c.gmm

if [[ -e ./src/lab2_vit ]] ; then
    binStr="./src/lab2_vit"
else
    echo "Couldn't find program to execute."
    exit 1
fi

echo "Decoding ..."
$binStr --gmm p4b.300c.gmm --audio_file p018k7tc.100.dat \
    --graph_file p018k1.loop.fsm --word_syms p018k2.syms \
    --dcd_file p4b.300c.dcd

echo "Computing WER ..."
#p018h1.calc-wer.sh p4b.300c.dcd p018k7tc.100.trn temp
./calc-wer.sh p018k7t.100.trn p4b.300c.dcd > p4b.300c.dcd.wer

