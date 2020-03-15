# If you are using cbtool with kind, suffix should be _kind
SUF="$1"
MONGO_CONFIG="mongodb${SUF}.conf"
echo "Script first argument is an suffix: ${SUF}"
echo "Mongodb configs: ${MONGO_CONFIG}"

sudo redis-server /etc/redis/redis.conf &
sudo cp $MONGO_CONFIG /etc/mongodb.conf &&
sudo mongod --config /etc/mongodb.conf &
