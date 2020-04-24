#!/bin/bash
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <expfiles>"
  exit 1
fi

for kEXPFILE in $@; do
  ./clear_influxdb.sh
  kEXPFILE="traces/$kEXPFILE"
  kIMAGES=`awk '/^aiattach/ {print $2}' ../$kEXPFILE | sort -u`
  for kIMAGE in $kIMAGES; do
    echo "Pull image $kIMAGE"
    ./pull_images.sh $kIMAGE
  done
  cd .. || exit 1
  echo "Will run cbtool for expfile: $kEXPFILE"
  kEXPID=`awk '/^expid/ {print}' $kEXPFILE | sed 's/expid //'`
  echo "Detected expid : $kEXPID"
  sudo ./cb --soft_reset --trace $kEXPFILE
  cd -
  ./export_exp.sh $kEXPID
done

