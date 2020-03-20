if [[ $# -ne 1 ]]; then
	echo "Usage $0 <expid>"
	exit 1
fi
kEXPID=$1

echo "Copy exp data for $kEXPID"
DIR=/home/pziecik/cbtool/data/$kEXPID
RESOURCES_DIR=/home/pziecik/cbtool/myscripts/resources
HOST="baati"
scp -o 'ProxyJump students' -r $HOST:$DIR . || exit 1
scp -o 'ProxyJump students' -r $HOST:$RESOURCES_DIR $kEXPID || exit 1
cd $kEXPID
gnome-open 008_vm_app_throughput_vs_time_plot_redis_ycsb.pdf &
