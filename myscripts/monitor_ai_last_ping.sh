LAST_AI=$1
TMP_FILE="monitor_ai_last_ping_tmp.txt"
rm $TMP_FILE
touch $TMP_FILE
for i in $(eval echo {1..$LAST_AI}); do
	T=$(cat /var/log/cloudbench/root_remotescripts.log | grep "ai-$i" | tail -n 1 | awk '{print $3}')
	echo "$T ai-$i" >> $TMP_FILE
done
sort $TMP_FILE -o $TMP_FILE
head -1 $TMP_FILE
tail -n 1 $TMP_FILE

wc -l $TMP_FILE
