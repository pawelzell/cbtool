#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <expid>"
  exit 1
fi
kEXPID=$1

./export_resource_metrics.sh || exit 1

cd .. &&
cp -r myscripts/resources data/$kEXPID

