kDIR="cleanup"
kINFLUX_POD=`kubectl get pods | awk '/^influx/ {print $1}'`
echo "Will clear influxdb database for pod: $kINFLUX_POD"

kubectl exec --namespace default influxdb-68999c6797-5bqnq -- bash -c "mkdir -p $kDIR"
kubectl cp helper_clear_influxdb.sh $kINFLUX_POD:$kDIR #&&
kubectl exec --namespace default $kINFLUX_POD -- bash -c "cd $kDIR && ./helper_clear_influxdb.sh"
