################################################################################
##
## Copyright (c) 2018 ROCm Developer Tools
##
## MIT LICENSE:
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

set(MAKE_CMD "${CMAKE_SOURCE_DIR}/regression/make_ctest_conf_logging.py" )
include(tests_conf_group_logging)

set(RVS_TEST_GROUP "usr")
rvs_register_test_group_logging(${RVS_TEST_GROUP})
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp_${RVS_TEST_GROUP}.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf_${RVS_TEST_GROUP}.cmake)

set(RVS_TEST_GROUP "kernel")
rvs_register_test_group_logging(${RVS_TEST_GROUP})
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp_${RVS_TEST_GROUP}.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf_${RVS_TEST_GROUP}.cmake)

set(RVS_TEST_GROUP "pkg")
rvs_register_test_group_logging(${RVS_TEST_GROUP})
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp_${RVS_TEST_GROUP}.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf_${RVS_TEST_GROUP}.cmake)

set(RVS_TEST_GROUP "ldchk")
rvs_register_test_group_logging(${RVS_TEST_GROUP})
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp_${RVS_TEST_GROUP}.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf_${RVS_TEST_GROUP}.cmake)

set(RVS_TEST_GROUP "fc")
rvs_register_test_group_logging(${RVS_TEST_GROUP})
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp_${RVS_TEST_GROUP}.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf_${RVS_TEST_GROUP}.cmake)

