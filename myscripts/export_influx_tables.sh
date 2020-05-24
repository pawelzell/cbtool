kDB='type_aware_scheduler'

# Export all metrics for pods from default namespace or for whole node
kREG="^metric\/(pod\/default\/|node|os)[[:graph:]]{,}$"
TABLES=$(influx -database $kDB -execute 'SHOW MEASUREMENTS')

rm *.csv
rm -rf resources
mkdir resources
for TABLE in $TABLES; do
  if [[ $TABLE =~ $kREG ]]; then
    echo $TABLE
    TABLE_OUT="$(echo $TABLE | sed 's/\//_/g').csv"
    influx -database $kDB -execute "SELECT * FROM \"${TABLE}\"" -format csv > ${TABLE_OUT}
    cp ${TABLE_OUT} resources
  fi
done

