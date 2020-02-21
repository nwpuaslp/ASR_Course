#!/bin/bash -e


if [[ -e ./lab3_lm ]] ; then
    binStr="./lab3_lm"
elif [[ -e Lab3Lm.class ]] ; then
    binStr="java Lab3Lm"
else
    echo "Couldn't find program to execute."
    exit 1
fi


for n in 1 2 3 ; do
$binStr --vocab lab3.syms --train minitrain.txt --test test1.txt \
    --n $n --word_probs p3a.$n.probs
done


