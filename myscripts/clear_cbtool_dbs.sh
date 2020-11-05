if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ip>"
  exit 1
fi
IP=$1
redis-cli -h $IP -p 6380 FLUSHALL
mongo ${IP}:27018/metrics flush_mongodb.js
