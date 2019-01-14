#!/bin/bash


export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`
echo ${RVS_BATCH_UTC}" 1. batch started  UID "${RVS_UID}  > $RVS_BATCH_SCRIPTS/cron.log

# run Ubuntu batch builds
export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 2. run ubuntu " >> $RVS_BATCH_SCRIPTS/cron.log
cd ${RVS_BATCH_SCRIPTS}
${RVS_BATCH_SCRIPTS}/rvs_ubuntu.sh


# runc CentOS Docker image and test script
export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 3. run centos " >> $RVS_BATCH_SCRIPTS/cron.log

cd ${RVS_BATCH_SCRIPTS}
${RVS_BATCH_SCRIPTS}/rvs_centos.sh

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 4. batch finished "  >> $RVS_BATCH_SCRIPTS/cron.log
