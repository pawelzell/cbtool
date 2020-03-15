if [[ $# -ne 1 ]]; then
	echo "Illegal number of parameters, got $# expected 1"
	exit 1
fi

echo "Copy exp data for $1"
DIR=/home/pziecik/cbtool/data/$1
HOST="baati"
scp -o 'ProxyJump students' -r $HOST:$DIR . || exit 1
cd $1
gnome-open 008_vm_app_throughput_vs_time_plot_redis_ycsb.pdf &
