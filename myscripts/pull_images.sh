if [[ $# -lt 1 ]]; then
  echo "usage $0 <images>"
  echo "example: $0 ycsb giraph"
  exit 1
fi
kVER_SUF="-amd64"
# ubuntu_cb_open_daytrader - do not pull, image from docker hub is broken. 
# Fixed image should be present on baati

for IMAGE in $@; do
	if [ ${IMAGE: -4} == "ycsb" ]; then
		IMAGE="ycsb"
	fi
	if [[ $IMAGE == "linpack" ]]; then
		sudo docker load < linpack.tar
	else
		IMAGE="ibmcb/ubuntu_cb_${IMAGE}"
		IMAGE_FULL="${IMAGE}${kVER_SUF}"
		echo "pull image ${IMAGE_FULL}"
		sudo docker pull $IMAGE_FULL:master
		sudo docker tag $IMAGE_FULL:master $IMAGE
	fi
done
