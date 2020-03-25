if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <influxdb_pod>"
  exit 1
fi
kINFLUX_POD=$1
echo "Will clear influxdb database for pod: $kINFLUX_POD"

kubectl cp helper_clear_influxdb.sh $kINFLUX_POD: &&
kubectl exec --namespace default $kINFLUX_POD -- bash -c "./helper_clear_influxdb.sh"
