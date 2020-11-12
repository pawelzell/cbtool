#!/bin/bash
if [[ (( $# -lt 2 ) || ( $1 != "-files" )) && (( $# -ne 3 ) || ( $1 != "-scheduler" )) ]]; then
  echo "Usage: $0 -files <expfiles from ../traces dir>"
  echo "Usage: $0 -scheduler <startid> <endid>"
  exit 1
fi

pull_images () {
  kIMAGES=`awk '/^aiattach/ {print $2}' ../${kEXPFILE} | sort -u`
  for kIMAGE in $kIMAGES; do
    echo "Pull image $kIMAGE"
    ./pull_images.sh $kIMAGE
  done
}

kEXPDIR="../traces"
if [[ $1 == "-files" ]]; then
  kEXPFILES=${@:2}
else
  kEXPFILES=()
  for i in $(eval echo {$2..$3}); do
    for j in {0..0}; do
        kEXPFILES+=("${i}scheduler${j}_custom")
        kEXPFILES+=("${i}scheduler${j}_default")
    done
  done
fi

for kEXPFILE in ${kEXPFILES[@]}; do
  kEXPFILE="myscripts/$kEXPDIR/$kEXPFILE"
  ./export_config_to_scheduler.sh ../${kEXPFILE} type-aware-scheduler
  ./clear_influxdb.sh
  echo "Sleep for 15s to make sure that scheduler read the experiment config"
  sleep 15s
  # When abstraction=deployemnt is used there is no need to pull images
  #pull_images
  cd .. || exit 1
  echo "Will run cbtool for expfile: $kEXPFILE"
  kEXPID=`awk '/^expid/ {print}' ${kEXPFILE} | sed 's/expid //'`
  echo "Detected expid : $kEXPID"
  sudo ./cb --soft_reset --trace ${kEXPFILE}
  sudo chown $USER:$USER data/${kEXPID}
  cd -
  ./export_exp.sh ${kEXPID}
done

