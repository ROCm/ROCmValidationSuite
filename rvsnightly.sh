#!/bin/bash
cd /work/rvsnightly
rm -rf build
rm -rf ROCmValidationSuite
ctest -S rvs_ctest_nightly.cmake
