#!/bin/bash
cd /work/rvsnightly
if [ -d build/Testing ]
then
    mv build/Testing /work/rvsnightly/Testing
fi
rm -rf build
rm -rf ROCmValidationSuite
mkdir build
if [ -d /work/rvsnightly/Testing ]
then
    mv /work/rvsnightly/Testing build/Testing
fi

ctest -DRVS_BRANCH:STRING=develop -S rvs_ctest_nightly.cmake

cd /work/rvsnightly
if [ -d build/Testing ]
then
    mv build/Testing /work/rvsnightly/Testing
fi
rm -rf build
rm -rf ROCmValidationSuite
mkdir build
if [ -d /work/rvsnightly/Testing ]
then
    mv /work/rvsnightly/Testing build/Testing
fi

ctest -DRVS_BRANCH:STRING=master -S rvs_ctest_nightly.cmake
