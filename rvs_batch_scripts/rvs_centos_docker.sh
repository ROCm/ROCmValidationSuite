#!/bin/bash

#sudo docker run --privileged=true -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /work/rvs_batch_scripts:/work/batch_scripts -v /home/user1:/home/root d4fb34eba628 /work/nightly/centos.sh


# since we are now in docker image, we need to define vars again
export RVS_CTEST_BUILD_TYPE=Nightly
export RVS_HOST="CentOS 7"
export RVS_BATCH_SCRIPTS=/work/batch_scripts
export RVS_BATCH_BUILD=/work/batch_build
export RVS_WB=${RVS_BATCH_BUILD}/centos

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`

echo ${RVS_BATCH_UTC}" 1. CentOS Docker script starting   UID "${RVS_UID} > $RVS_BATCH_SCRIPTS/centos.log

# we must source scl in order to have the right toolchain
source scl_source enable devtoolset-7

mkdir -p $RVS_WB
cd $RVS_WB

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 2. before ctest develop " >> $RVS_BATCH_SCRIPTS/centos.log

ctest \
-DRVS_BRANCH:STRING=develop \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=TRUE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 3. before ctest master " >> $RVS_BATCH_SCRIPTS/centos.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 4. before ctest master w. local rocBLAS" >> $RVS_BATCH_SCRIPTS/centos.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Release -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=FALSE -DWITH_TESTING:BOOL=FALSE \
-DRVS_ROCBLAS=1 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 5. done." >> $RVS_BATCH_SCRIPTS/centos.log
echo "" >> $RVS_BATCH_SCRIPTS/centos.log

exit 0
