################################################################################
##
## Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

set(ROCBLAS_LIB "rocblas")
set(HIPRAND_LIB "hiprand")
set(ROC_THUNK_NAME "hsakmt")
set(CORE_RUNTIME_NAME "hsa-runtime")
set(CORE_RUNTIME_TARGET "${CORE_RUNTIME_NAME}64")

set(UT_LINK_LIBS  libpthread.so libpci.so libm.so libdl.so "lib${ROCM_SMI_LIB}.so"
  ${ROCBLAS_LIB} ${ROC_THUNK_NAME} ${CORE_RUNTIME_TARGET} ${ROCM_CORE} ${YAML_CPP_LIBRARIES} ${HIPRAND_LIB}
)

# Add directories to look for library files to link
link_directories(${ROCM_SMI_LIB_DIR} ${ROCT_LIB_DIR} ${ROCBLAS_LIB_DIR} ${HIPRAND_LIB_DIR})

set (UT_SOURCES src/action.cpp test/unitsmqt.cpp
)

# add unit tests
include(tests_unit)

# Add configuration tests
include(tests_conf_logging)

