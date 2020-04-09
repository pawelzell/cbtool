if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <expfiles>"
  exit 1
fi

for kEXPFILE in $@; do
  ./clear_influxdb.sh
  cd .. || exit 1
  kEXPFILE="traces/$kEXPFILE"
  echo "Will run cbtool for expfile: $kEXPFILE"
  kEXPID=`awk '/^expid/ {print}' $kEXPFILE | sed 's/expid //'`
  echo "Detected exdbpid : $kEXPID"
  ./cb --soft_reset --trace $kEXPFILE
  cd -
  ./export_exp.sh $kEXPID
done

