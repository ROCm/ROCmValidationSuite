# ################################################################################
# #
# # Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
# #
# # MIT LICENSE:
# # Permission is hereby granted, free of charge, to any person obtaining a copy of
# # this software and associated documentation files (the "Software"), to deal in
# # the Software without restriction, including without limitation the rights to
# # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# # of the Software, and to permit persons to whom the Software is furnished to do
# # so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# #
# ###############################################################################

#!/bin/sh
echo 'mix_gmgst1';sudo ../../../bin/rvs -c ../conf/mix_gmgst1.conf -d 3;  date
echo 'mix_gmgst2';sudo ../../../bin/rvs -c ../conf/mix_gmgst2.conf -d 3; date
echo 'mix_gmgst3';sudo ../../../bin/rvs -c ../conf/mix_gmgst3.conf -d 3; date
echo 'mix_gmiet';sudo ../../../bin/rvs -c ../conf/mix_gmiet.conf -d 3; date
echo 'mix_gmiet1';sudo ../../../bin/rvs -c ../conf/mix_gmiet1.conf -d 3; date
echo 'mix_gmpebb1';sudo ../../../bin/rvs -c ../conf/mix_gmpebb1.conf -d 3; date
echo 'mix_gmpebb2';sudo ../../../bin/rvs -c ../conf/mix_gmpebb2.conf -d 3; date
echo 'mix_gmpebb3';sudo ../../../bin/rvs -c ../conf/mix_gmpebb3.conf -d 3; date
echo 'mix_gmpebb4';sudo ../../../bin/rvs -c ../conf/mix_gmpebb4.conf -d 3; date
