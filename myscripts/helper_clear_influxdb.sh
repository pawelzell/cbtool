kDB='type_aware_scheduler'
influx -execute "drop database $kDB" && influx -execute "create database $kDB"
