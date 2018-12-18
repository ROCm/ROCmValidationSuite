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


set(MAKE_CMD "${CMAKE_CURRENT_SOURCE_DIR}/../regression/make_ctest_conf.py" )

execute_process(COMMAND ${MAKE_CMD} ttp ${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp.cmake "${RVS}*.conf" ${RVS}
  RESULT_VARIABLE RVS_EP_STS
  ERROR_VARIABLE RVS_EP_ERROR
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if (RVS_EP_STS)
  MESSAGE("RVS_EP_STS: ${RVS_EP_STS} ")
  MESSAGE("RVS_EP_ERROR: ${RVS_EP_ERROR} ")
endif()

execute_process(COMMAND ${MAKE_CMD} ttf ${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf.cmake "ttf_${RVS}*.conf" ${RVS}
  RESULT_VARIABLE RVS_EP_STS
  ERROR_VARIABLE RVS_EP_ERROR
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if (RVS_EP_STS)
  MESSAGE("RVS_EP_STS: ${RVS_EP_STS} ")
  MESSAGE("RVS_EP_ERROR: ${RVS_EP_ERROR} ")
endif()

include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttp.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/tests_conf_ttf.cmake)
