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

# generate random test files

## generate conf files
#MESSAGE("RVS PATH: ${CMAKE_CURRENT_SOURCE_DIR}")
set(MAKE_CMD "${CMAKE_CURRENT_SOURCE_DIR}/../regression/make_pqt_conf.py")
#MESSAGE("COMMAND: ${MAKE_CMD}")
execute_process(COMMAND ${MAKE_CMD})

# include resulting .cmake file with random tests declarations
include(${CMAKE_CURRENT_SOURCE_DIR}/rand_tests.cmake)

include(tests_conf_logging)

add_test(NAME pqt.coverage.1 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../regression/run_and_check_test.py ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../rvs/conf/pqt_coverage1.conf true true false ttf 3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME pqt.coverage.2 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../regression/run_and_check_test.py ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../rvs/conf/pqt_coverage2.conf true true false ttp 3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME pqt.coverage.3 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../regression/run_and_check_test.py ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../rvs/conf/pqt_coverage3.conf true true false ttp 3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME pqt.coverage.4 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../regression/run_and_check_test.py ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../rvs/conf/pqt_coverage4.conf true true false ttf 3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME pqt.coverage.5 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../regression/run_and_check_test.py ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../rvs/conf/pqt_coverage5.conf true true false ttf 3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME pqt.coverage.6 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../regression/run_and_check_test.py ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../rvs/conf/pqt_coverage6.conf true true false ttf 3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)