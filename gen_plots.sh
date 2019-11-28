for exp in $(ls data); do
	echo $exp
	cd "data/$exp"; sudo ./plot.sh; cd ../..
	cp -r "data/$exp" "plots/$exp"
done
