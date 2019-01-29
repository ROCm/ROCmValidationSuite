#!/bin/bash

export RVS_CTEST_BUILD_TYPE=Experimental
export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_BUILD=/work/tst_batch_build

export RVS_HOST="Ubuntu 16.04"
export RVS_WB=${RVS_BATCH_BUILD}/ubuntu
mkdir -p ${RVS_WB}
cd ${RVS_WB}

echo "RVS_CTEST_BUILD_TYPE=${RVS_CTEST_BUILD_TYPE}">$RVS_BATCH_SCRIPTS/single.log

# build and test branch develop
rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`

mkdir -p ${RVS_WB}/build

echo ${RVS_BATCH_UTC}" 1. before ctest UID: "${RVS_UID} >> $RVS_BATCH_SCRIPTS/single.log

ctest -DRVS_TAG=" EXP " \
-DRVS_BRANCH:STRING=develop \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=TRUE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 2. after ctest " >> $RVS_BATCH_SCRIPTS/single.log

