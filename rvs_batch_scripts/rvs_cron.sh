#!/bin/bash


export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_BUILD=/work/rvs_batch_build

export RVS_BATCH_UTC=`date -u`
echo " 1. batch started "${RVS_BATCH_UTC} > $RVS_BATCH_SCRIPTS/cron.log
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
docker run --privileged=true --rm -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /work/rvs_batch_scripts:/work/batch_scripts -v /home/user1:/home/root 88e8c5704c2e /work/batch_scripts/rvs_centos.sh  >> $RVS_BATCH_SCRIPTS/cron.log

export RVS_BATCH_UTC=`date -u`
echo " 4. done "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/cron.log
