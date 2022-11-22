# ################################################################################
# #
# # Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
echo "=========================Starting RVS Suite========================="

echo "===========================gpup========================="
sudo ./gpup.new.sh 2>&1 | tee gpup.txt
echo "===========================gst========================="
sudo ./gst.new.sh  2>&1 | tee ggst.txt
echo "===========================iet========================="
sudo ./iet.new.sh  2>&1 | tee iet.txt
echo "===========================pebb========================="
sudo ./pebb.new.sh  2>&1 | tee pebb.txt
echo "===========================peqt========================="
sudo ./peqt.new.sh  2>&1 | tee peqt.txt
echo "==========================pesm========================="
sudo  ./pesm.new.sh  2>&1 | tee pesm.txt
echo "===========================pbqt========================="
sudo ./pbqt.new.sh  2>&1 | tee pbqt.txt
echo "===========================memory========================="
sudo ./rvs-mem.sh  2>&1 | tee memory.txt

echo "=========================RVS Suite completed, check txt for output============"
