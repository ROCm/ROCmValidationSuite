#!/bin/bash


export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_BUILD=/work/rvs_batch_build
export RVS_BATCH_UTC=`date -u`

rm -rf ${RVS_BATCH_BUILD}
mkdir -p ${RVS_BATCH_BUILD}

# run Ubuntu batch builds
cd ${RVS_BATCH_SCRIPTS}
${RVS_BATCH_SCRIPTS}/rvs_ubuntu.sh

# runc CentOS Docker image and test script
docker run --privileged=true -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /work/rvs_batch_scripts:/work/batch_scripts -v /home/user1:/home/root d4fb34eba628 /work/batch_scripts/rvs_centos.sh
