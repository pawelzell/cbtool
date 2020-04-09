kDIR="resources"
kINFLUX_POD=`kubectl get pods | awk '/^influx/ {print $1}'`
echo "Will try to export metrics from influxdb pod: $kINFLUX_POD"

rm -rf $kDIR
kubectl exec --namespace default $kINFLUX_POD -- bash -c "mkdir -p $kDIR" &&
kubectl cp export_influx_tables.sh $kINFLUX_POD:/$kDIR &&
kubectl exec --namespace default $kINFLUX_POD -- bash -c "cd $kDIR && ./export_influx_tables.sh" &&
kubectl cp $kINFLUX_POD:/$kDIR $kDIR #&&

