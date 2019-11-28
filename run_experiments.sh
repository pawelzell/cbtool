for file in $(ls traces/); do
	echo "Running experiment $file"
	sudo ./cb --soft_reset --trace traces/$file
done

