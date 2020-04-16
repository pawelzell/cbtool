if [[ $# -ne 1 ]]; then
	echo "Usage $0 <remotehost>"
	exit 1
fi
kREMOTEUSER=pziecik
kREMOTEHOST=$1

kLOCAL_DIR="data"
kTMP_DIR="../data"

kSCRIPT_BASE_DIR=$PWD
kREMOTE_DIR="/home/$kREMOTEUSER/cbtool/data"
kEXPS=`ssh -J students $kREMOTEHOST ls /home/$kREMOTEUSER/cbtool/data`
for kEXP in $kEXPS; do
  if [[ ! -e $kLOCAL_DIR/$kEXP ]]; then
		echo "Copy $kEXP"
		# 1 Copy
		kDIR="$kREMOTE_DIR/$kEXP"
		scp -o 'ProxyJump students' -r $kREMOTEHOST:$kDIR $kTMP_DIR

		# 2 Plot
		cd $kTMP_DIR/$kEXP
		kRES="$?"
		if [[ $kRES -ne 0 ]]; then
			echo "ASSERT cd $kTMP_DIR/$kEXP - FAILED"
			exit 1
		fi 
		./plot.sh
		cd $kSCRIPT_BASE_DIR
		if [[ $PWD != $kSCRIPT_BASE_DIR ]]; then
			echo "ASSERT UNEXPECTED DIR $PWD exitting" 
			exit 1
		fi 

		# 3 Move
		mv $kTMP_DIR/$kEXP $kLOCAL_DIR
	else
		echo "Skipping $kEXP - directory already exists"
	fi
done



