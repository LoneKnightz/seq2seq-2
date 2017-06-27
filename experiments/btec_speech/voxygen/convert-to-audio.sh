#!/bin/bash
set -e

filename=$1
dir=$2
voice=$3   # Agnes, Fabienne, Helene, Loic, Marion, Michel, Philippe  (default = Agnes)

mkdir -p ${dir}
lines=`wc -l ${filename} | cut -d' ' -f1`
digits=$((`echo ${lines} | wc -c` - 1))

for i in `seq 1 ${lines}`;
do
    num=`printf "%0${digits}d" ${i}`
    cat ${filename} | sed -n "${i},${i}p" > ${dir}/${num}.txt
    experiments/speech/voxygen/wsclient.py -i ${dir}/${num}.txt -o ${dir}/${num}.wav header=wav-header frequency=16000 coding=lin voice=${voice}
    rm ${dir}/${num}.txt
done
