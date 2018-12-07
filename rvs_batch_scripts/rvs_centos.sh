#!/bin/bash

#sudo docker run --privileged=true -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /work/rvs_batch_scripts:/work/batch_scripts -v /home/user1:/home/root d4fb34eba628 /work/nightly/centos.sh

# since we are now in docker image, we need to define vars again
export RVS_BATCH_SCRIPTS=/work/batch_scripts
export RVS_BATCH_BUILD=/work/batch_build
export RVS_WB=${RVS_BATCH_BUILD}/centos

export RVS_BATCH_UTC=`date -u`
echo " 1. before sourcing "${RVS_BATCH_UTC} > $RVS_BATCH_SCRIPTS/centos.log
source scl_source enable devtoolset-7

echo " 2. before creating "${RVS_WB} >> $RVS_BATCH_SCRIPTS/centos.log
mkdir -p $RVS_WB
cd $RVS_WB

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo " 3. before ctest develop "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log

ctest -DRVS_BRANCH:STRING=develop -DRVS_HOST:STRING="CentOS 7" -S $RVS_BATCH_SCRIPTS/rvs_ctest_nightly.cmake

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo " 4. before ctest master "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log
ctest -DRVS_BRANCH:STRING=master -DRVS_HOST:STRING="CentOS 7" -S $RVS_BATCH_SCRIPTS/rvs_ctest_nightly.cmake

export RVS_BATCH_UTC=`date -u`
echo " 5. done. time is "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log

exit 0
