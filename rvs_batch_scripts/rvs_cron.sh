#!/bin/bash


export RVS_CTEST_BUILD_TYPE=Nightly
export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_BUILD=/work/rvs_batch_build

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`
echo " 1. batch started "${RVS_BATCH_UTC}"  "${RVS_UID}  > $RVS_BATCH_SCRIPTS/cron.log
rm -rf ${RVS_BATCH_BUILD}
mkdir -p ${RVS_BATCH_BUILD}

# run Ubuntu batch builds
export RVS_BATCH_UTC=`date -u`
echo " 2. run ubuntu "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/cron.log
cd ${RVS_BATCH_SCRIPTS}
${RVS_BATCH_SCRIPTS}/rvs_ubuntu.sh


# runc CentOS Docker image and test script
export RVS_BATCH_UTC=`date -u`
echo " 3. run centos "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/cron.log

strcmd="docker run --privileged=true --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /home/user1:/home/root -v /work/rvs_batch_scripts:/work/batch_scripts 88e8c5704c2e /work/batch_scripts/rvs_centos.sh 2>>$RVS_BATCH_SCRIPTS/cron.log"

echo $strcmd >> $RVS_BATCH_SCRIPTS/cron.log

eval  $strcmd

errcode=$?

if [ $errcode != 0 ]; then
  echo "Error: " $errcode >> $RVS_BATCH_SCRIPTS/cron.log
fi

export RVS_BATCH_UTC=`date -u`
echo " 4. done "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/cron.log
