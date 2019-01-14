#!/bin/bash


export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`
echo ${RVS_BATCH_UTC}" - CentOS batch started   UID "${RVS_UID}  > $RVS_BATCH_SCRIPTS/centos.log

strcmd="docker run --privileged=true --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /home/user1:/home/root -v /work/rvs_batch_scripts:/work/batch_scripts 88e8c5704c2e /work/batch_scripts/rvs_centos_docker.sh 2>>$RVS_BATCH_SCRIPTS/centos.log"

echo $strcmd >> $RVS_BATCH_SCRIPTS/centos.log

eval  $strcmd

errcode=$?

if [ $errcode != 0 ]; then
  echo "Error: " $errcode >> $RVS_BATCH_SCRIPTS/centos.log
fi

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" - CentOS batch finsihed " >> $RVS_BATCH_SCRIPTS/centos.log
