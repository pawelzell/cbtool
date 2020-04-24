#!/bin/bash
kDIR="data"
kIMAGES="redis_ycsb wrk hadoop linpack"
kMETRICS=( latency throughput )
kPREFIXES=( 007 008 )

cd $kDIR
kEXPIDS=`ls`
for kEXPID in $kEXPIDS; do
	for kIMAGE in $kIMAGES; do 
		for idx in ${!kMETRICS[@]}; do
			kMETRIC=${kMETRICS[$idx]}
			kPREFIX=${kPREFIXES[$idx]}

		  kIN="$kEXPID/${kPREFIX}_vm_app_${kMETRIC}_vs_time_plot_${kIMAGE}.pdf"
		  kOUT="${kEXPID}_${kMETRIC}_${kIMAGE}.pdf"
			if [[ -e $kIN ]]; then
				echo "Copy plot to $kDIR/$kOUT"
				cp $kIN $kOUT
			fi
		done
	done
done






