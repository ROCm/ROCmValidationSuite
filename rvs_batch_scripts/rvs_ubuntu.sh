#!/bin/bash

export RVS_HOST="Ubuntu 16.04"
export RVS_WB=${RVS_BATCH_BUILD}/ubuntu
mkdir -p ${RVS_WB}

# build and test branch develop
cd ${RVS_WB}
if [ -d ${RVS_WB}/build/Testing ]
then
    mv ${RVS_WB}/build/Testing ${RVS_WB}/Testing
fi

rm -rf build
rm -rf ROCmValidationSuite
mkdir -p ${RVS_WB}/build

if [ -d ${RVS_WB}/Testing ]
then
    mv ${RVS_WB}/Testing build/Testing
fi

ctest -DRVS_BRANCH:STRING=iss330 -DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

# build and test branch master

cd ${RVS_WB}
if [ -d ${RVS_WB}/build/Testing ]
then
    mv ${RVS_WB}/build/Testing ${RVS_WB}/Testing
fi

rm -rf build
rm -rf ROCmValidationSuite
mkdir -p ${RVS_WB}/build

if [ -d ${RVS_WB}/Testing ]
then
    mv ${RVS_WB}/Testing build/Testing
fi

ctest -DRVS_BRANCH:STRING=master -DRVS_HOST:STRING="${RVS_HOST}" -S ${RVS_BATCH_SCRIPTS}/rvs_ctest_nightly.cmake

