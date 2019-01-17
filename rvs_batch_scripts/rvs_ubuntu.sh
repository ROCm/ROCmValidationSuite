#!/bin/bash


export RVS_CTEST_BUILD_TYPE=Nightly
export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_BUILD=/work/rvs_batch_build
export RVS_HOST="Ubuntu 16.04"
export RVS_WB=${RVS_BATCH_BUILD}/ubuntu
mkdir -p ${RVS_WB}
cd ${RVS_WB}

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`
echo "RVS_CTEST_BUILD_TYPE=${RVS_CTEST_BUILD_TYPE}">$RVS_BATCH_SCRIPTS/ubuntu.log
echo "RVS_UID=${RVS_UID}">>$RVS_BATCH_SCRIPTS/ubuntu.log

# build and test branch develop
rm -rf build
rm -rf ROCmValidationSuite
#mkdir -p ${RVS_WB}/build

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 1. before ctest develop " >> $RVS_BATCH_SCRIPTS/ubuntu.log

ctest \
-DRVS_BRANCH:STRING=develop \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=TRUE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake


# build and test branch master
rm -rf build
rm -rf ROCmValidationSuite
#mkdir -p ${RVS_WB}/build

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 2. before ctest master " >> $RVS_BATCH_SCRIPTS/ubuntu.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake


# build branch master with local rocBLAS
rm -rf build
rm -rf ROCmValidationSuite
#mkdir -p ${RVS_WB}/build

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 3. before ctest master w. local rocBLAS" >> $RVS_BATCH_SCRIPTS/ubuntu.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Release -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=FALSE -DWITH_TESTING:BOOL=FALSE \
-DRVS_ROCBLAS=1 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

export RVS_BATCH_UTC=`date -u`
echo ${RVS_BATCH_UTC}" 4. done." >> $RVS_BATCH_SCRIPTS/ubuntu.log
echo "" >> $RVS_BATCH_SCRIPTS/ubuntu.log
