if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <expid> <influxdb_pod>"
  exit 1
fi
kEXPID=$1
kINFLUX_POD=$2
kDIR="resources"

./export_resource_metrics.sh $kINFLUX_POD || exit 1

cd .. &&
cp -r myscripts/resources data/$kEXPID &&
cd data/$kEXPID && ./plot.sh

