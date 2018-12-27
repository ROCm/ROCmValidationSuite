#!/bin/bash

export RVS_HOST="Ubuntu 16.04"
export RVS_WB=${RVS_BATCH_BUILD}/ubuntu
mkdir -p ${RVS_WB}
cd ${RVS_WB}

echo "RVS_CTEST_BUILD_TYPE=${RVS_CTEST_BUILD_TYPE}">>$RVS_BATCH_SCRIPTS/ubuntu.log
# build and test branch develop
rm -rf build
rm -rf ROCmValidationSuite

export RVS_BATCH_UTC=`date -u`
export RVS_UID=`id -u`:`id -g`
echo " 21. before creating "${RVS_WB} > $RVS_BATCH_SCRIPTS/ubuntu.log

mkdir -p ${RVS_WB}/build

export RVS_BATCH_UTC=`date -u`
echo " 22a. before ctest develop "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/ubuntu.log

ctest \
-DRVS_BRANCH:STRING=develop \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=TRUE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake


# build and test branch master
rm -rf build
rm -rf ROCmValidationSuite
mkdir -p ${RVS_WB}/build

export RVS_BATCH_UTC=`date -u`
echo " 22b. before ctest master "${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/ubuntu.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Debug -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=TRUE -DWITH_TESTING:BOOL=TRUE \
-DRVS_ROCBLAS=0 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake


# build branch master with local rocBLAS
rm -rf build
rm -rf ROCmValidationSuite
mkdir -p ${RVS_WB}/build

export RVS_BATCH_UTC=`date -u`
echo " 22c. before ctest master w. local rocBLAS"${RVS_BATCH_UTC} >> $RVS_BATCH_SCRIPTS/ubuntu.log

ctest \
-DRVS_BRANCH:STRING=master \
-DCTEST_BUILD_CONFIGURATION=Release -DRVS_CTEST_BUILD_TYPE:STRING=${RVS_CTEST_BUILD_TYPE} \
-DRVS_COVERAGE:BOOL=FALSE -DRVS_BUILD_TESTS:BOOL=FALSE -DWITH_TESTING:BOOL=FALSE \
-DRVS_ROCBLAS=1 -DRVS_ROCMSMI=1 \
-DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

