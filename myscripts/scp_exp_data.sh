if [[ $# -ne 1 ]]; then
	echo "Illegal number of parameters, got $# expected 1"
	exit 1
fi

echo "Copy exp data for $1"
DIR=~/cbtool/data/$1
HOST="baati"
scp -o 'ProxyJump students' -r $HOST:$DIR .
