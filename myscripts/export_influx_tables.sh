kDB='type_aware_scheduler'
kREDIS="redis|ycsb"
KGIRAPH="giraphmaster|giraphslave"
kHADOOP="hadoopmaster|hadoopslave"
kSYSBENCH="sysbench|mysql"
kDAYTRADER="client_open_daytrader|geronimo|mysql"

kROLES="$kREDIS|$kGIRAPH|$kHADOOP|$kSYSBENCH|$kDAYTRADER"
kMACHINES="baati|naan"

kREG="^metric\/(pod\/default(\/[-[:alnum:]]{,}($kROLES)[-[:alnum:]]{,}){2}|node\/($kMACHINES))\/(cpu|memory)$"
TABLES=$(influx -database $kDB -execute 'SHOW MEASUREMENTS')
#TABLES="metric/node/baati/cpu metric/node/naan/memory metric/pod/default/cb-pziecik-mykub-vm1-ycsb-ai-1/cb-pziecik-mykub-vm1-ycsb-ai-1/cpu metric/pod/default/cb-pziecik-mykub-vm1-ycsb-ai-1/cb-pziecik-mykub-vm1-ycsb-ai-1/memory"

rm *.csv
for TABLE in $TABLES; do
  if [[ $TABLE =~ $kREG ]]; then
    echo $TABLE
    TABLE_OUT="$(echo $TABLE | sed 's/\//_/g').csv"
    influx -database $kDB -execute "SELECT * FROM \"${TABLE}\"" -format csv > ${TABLE_OUT}
  fi
done

