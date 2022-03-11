#!/bin/bash
out=$2
dir=$1

for line in `ls $dir`
do
	name=`basename ${line} .pssm`
	python script.py line "${2}${name}.feature"
done