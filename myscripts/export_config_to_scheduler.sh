#!/bin/bash
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <exp_file>"
  exit 1
fi

kSCHEDULER_POD=`kubectl get pods | awk '/^type-aware-scheduler/ {print $1}'`
echo "Will copy experiment config file to: $kSCHEDULER_POD"

kubectl cp $0 $kINFLUX_POD:/exp
