#!/bin/bash
kWORD="redis"
kREPLACE="redis_ycsb"

kDIRS=`ls | grep $kWORD`
for kDIR in $kDIRS; do
  kNEW_DIR=`echo $kDIR | sed "s/${kWORD}/${kREPLACE}/g"`
  echo "input dir: $kDIR output dir: $kNEW_DIR"
  mv $kDIR $kNEW_DIR
done
