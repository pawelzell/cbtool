kVER_SUF="-amd64"
IMAGES="ubuntu_cb_sysbench ubuntu_cb_hadoop ubuntu_cb_giraph ubuntu_cb_ycsb"
# ubuntu_cb_open_daytrader - do not pull, image from docker hub is broken. 
# Fixed image should be present on baati

for IMAGE in $IMAGES; do
	IMAGE="ibmcb/${IMAGE}"
	IMAGE_FULL="${IMAGE}${kVER_SUF}"
	echo "pull image ${IMAGE_FULL}"
	sudo docker pull $IMAGE_FULL:master
	sudo docker tag $IMAGE_FULL:master $IMAGE
done
