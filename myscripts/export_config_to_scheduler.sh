#!/bin/bash
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <exp_file> <scheduler>"
  exit 1
fi

kSCHEDULER="$2"
kSCHEDULER_POD=`kubectl get pods | awk '/^'"$kSCHEDULER"'/ {print $1}'`
echo "Will copy experiment config file $1 to: $kSCHEDULER_POD"

kubectl cp $1 $kSCHEDULER_POD:/exp
