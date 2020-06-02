#!/bin/bash
if [[ ( $# -lt 2 ) || (( $1 != "-files" ) && ( $1 != "-dir" )) ]]; then
  echo "Usage: $0 -files <expfiles from ../traces dir>"
  echo "Usage: $0 -dir <dir with expfiles>"
  exit 1
fi

if [[ $1 == "-files" ]]; then
  kEXPDIR="../traces"
  kEXPFILES=${@:2}
else
  kEXPDIR=$2
  kEXPFILES=`ls "$2"`
fi

for kEXPFILE in ${kEXPFILES}; do
  kEXPFILE="myscripts/$kEXPDIR/$kEXPFILE"
  ./export_config_to_scheduler.sh ../${kEXPFILE}
  ./clear_influxdb.sh
  kIMAGES=`awk '/^aiattach/ {print $2}' ../${kEXPFILE} | sort -u`
  for kIMAGE in $kIMAGES; do
    echo "Pull image $kIMAGE"
    ./pull_images.sh $kIMAGE
  done
  cd .. || exit 1
  echo "Will run cbtool for expfile: $kEXPFILE"
  kEXPID=`awk '/^expid/ {print}' ${kEXPFILE} | sed 's/expid //'`
  echo "Detected expid : $kEXPID"
  sudo ./cb --soft_reset --trace ${kEXPFILE}
  cd -
  ./export_exp.sh ${kEXPID}
done

