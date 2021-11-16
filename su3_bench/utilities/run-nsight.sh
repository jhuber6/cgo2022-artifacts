#!/bin/bash
set -x

exe=./bench_f32_cuda.exe
kernel=k_mat_nn
#threads=128
threads=4

filename=profile-${kernel}-t${threads}
skip=1
count=10
#sections="--section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SpeedOfLight --section SpeedOfLight_RooflineChart"
sections="--section SpeedOfLight --section SpeedOfLight_RooflineChart --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis"

srun nv-nsight-cu-cli -o $filename -k $kernel -s $skip -c $count $sections $exe -t $threads

