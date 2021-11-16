#!/bin/bash

iters=100
if [ $# -lt 2 ]; then
  printf "usage: %s executable out_file\n" $0
  exit 1
fi
exe=$1
outfile=$2

if [ -f $outfile ]; then
  echo "ERROR: $outfile already exists"
  exit 1
fi

for t in 36 64 72 128 144 256 288 512; do
  for i in {1..5}; do
    command="srun $exe -i $iters -t $t"
    echo $command >>$outfile
    $command 2>&1 >>$outfile
  done
done
