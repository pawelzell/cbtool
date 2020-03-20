kVER_SUF="-amd64"
IMAGES="ubuntu_cb_hadoop"
IMAGES="ubuntu_cb_hadoop ubuntu_giraph ubuntu_cb_ycsb"

for IMAGE in $IMAGES; do
	IMAGE="ibmcb/${IMAGE}"
	IMAGE_FULL="${IMAGE}${kVER_SUF}"
	echo "pull image ${IMAGE_FULL}"
	sudo docker pull $IMAGE_FULL:master
	sudo docker tag $IMAGE_FULL:master $IMAGE
done
