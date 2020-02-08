#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys


def usage() :
    print("convert_fmt.py : convert e6897 dcd fmt to evaluation fmt")
    print("         usage : python convert_fmt.py e6870_fmt eva_fmt")

if __name__ == '__main__':
    if len(sys.argv) == 1 :
        usage()
        sys.exit(0)
    dcd_file = sys.argv[1]
    eva_file = sys.argv[2]
    with open(eva_file,'w') as eva_f: 
        for line in open(dcd_file, 'r') :
            array = line.rstrip('\n').split()
            out_array = [array[-1]] + [ i  for i in array[:-1] if i !='~SIL']
            eva_f.write(' '.join(out_array) + '\n')