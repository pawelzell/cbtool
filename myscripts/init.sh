# If you are using cbtool with kind, suffix should be _kind
SUF=""

sudo redis-server /etc/redis/redis.conf &
sudo cp mongod${SUF}.conf /etc/mongodb.conf &&
sudo mongod --config /etc/mongodb.conf &
