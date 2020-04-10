kINFLUX_POD=`kubectl get pods | awk '/^influx/ {print $1}'`
echo "Will clear influxdb database for pod: $kINFLUX_POD"

kubectl cp helper_clear_influxdb.sh $kINFLUX_POD:/ #&&
kubectl exec --namespace default $kINFLUX_POD -- bash -c "cd / && ./helper_clear_influxdb.sh"
