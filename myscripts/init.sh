# If you are using cbtool with kind, suffix should be _kind
SUF=""

sudo redis-server /etc/redis/redis.conf &
sudo cp mongodb${SUF}.conf /etc/mongodb.conf &&
sudo mongod --config /etc/mongodb.conf &
