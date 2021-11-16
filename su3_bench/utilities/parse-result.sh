#!/bin/bash
if [ $# -lt 1 ]; then
  printf "usage: %s file\n" $0
  exit 1
fi

i=1
cat $1 | grep GFLOP | awk {'print $4'} | \
while read result; do
  echo -n ${result}
  j=$(( i % 5 ))
  if [ $j -eq 0 ]; then
    echo ""
  else
    echo -n ","
  fi
 ((  i += 1 ))
done

