#!/bin/bash -e


if [[ -e ./lab3_lm ]] ; then
    binStr="./lab3_lm"
elif [[ -e Lab3Lm.class ]] ; then
    binStr="java Lab3Lm"
else
    echo "Couldn't find program to execute."
    exit 1
fi


$binStr --vocab lab3.syms --train minitrain2.txt --test test1.txt \
    --count_file p1a.counts


