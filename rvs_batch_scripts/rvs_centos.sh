#!/bin/bash

#sudo docker run --privileged=true -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /work/rvs_batch_scripts:/work/batch_scripts -v /home/user1:/home/root d4fb34eba628 /work/nightly/centos.sh

# since we are now in docker image, we need to define vars again
export RVS_HOST="CentOS 7"
export RVS_BATCH_SCRIPTS=/work/batch_scripts
export RVS_BATCH_BUILD=/work/batch_build
export RVS_WB=${RVS_BATCH_BUILD}/centos

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`
echo " 31. before sourcing "${RVS_BATCH_UTC}"  "${RVS_UID} > $RVS_BATCH_SCRIPTS/centos.log
source scl_source enable devtoolset-7

echo " 32. before creating "${RVS_WB} >> $RVS_BATCH_SCRIPTS/centos.log
mkdir -p $RVS_WB
cd $RVS_WB

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo " 33a. before ctest develop "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log

ctest \
-DRVS_BRANCH:STRING=develop \
-DCTEST_BUILD_CONFIGURATION=Debug -DCTEST_BUILD_TYPE=Nightly \
-DRVS_COVERAGE:BOOL=TRUE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo " 33b. before ctest master "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Debug -DCTEST_BUILD_TYPE=Nightly \
-DRVS_COVERAGE:BOOL=TRUE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
echo " 33c. before ctest master w. local rocBLAS"${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log

ctest \
-DRVS_BRANCH:STRING=develop \
-DCTEST_BUILD_CONFIGURATION=Release -DCTEST_BUILD_TYPE=Nightly \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=FALSE -DWITH_TESTING:BOOL=FALSE \
-DRVS_ROCBLAS=1 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

export RVS_BATCH_UTC=`date -u`
echo " 34. done. time is "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/centos.log

exit 0
