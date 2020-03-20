if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <influxdb_pod>"
  exit 1
fi
kINFLUX_POD=$1
kDIR="resources"
echo "Will try to export metrics from influxdb pod: $kINFLUX_POD"

rm -rf $kDIR
kubectl exec --namespace default $kINFLUX_POD -- bash -c "mkdir -p $kDIR" &&
kubectl cp export_influx_tables.sh $kINFLUX_POD:/$kDIR &&
kubectl exec --namespace default $kINFLUX_POD -- bash -c "cd $kDIR && ./export_influx_tables.sh" &&
kubectl cp $kINFLUX_POD:/$kDIR $kDIR #&&
#kubectl exec --namespace default $kINFLUX_POD -- bash -c "influx -execute \'drop database type_aware_scheduler\' && influx -execute \'create database type_aware_scheduler\'"

