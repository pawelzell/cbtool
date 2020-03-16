IMAGES="ubuntu_cb_giraph"
#IMAGES="ubuntu_giraph ubuntu_cb_ycsb"

for IMAGE in $IMAGES; do
	IMAGE="ibmcb/${IMAGE}"
	echo "pull image ${IMAGE}"
	sudo docker pull $IMAGE:master
	sudo docker tag $IMAGE:master $IMAGE:latest
done
