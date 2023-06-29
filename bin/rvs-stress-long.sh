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
if [ "$1" == "-h" ] ; then
        echo
        echo "Usage: ./`basename $0` 3 (for 3 hours) or ./`basename $0` 6 (for 6 hours), default duration is 1 hour"
        echo
    exit 0
fi
if [ "$#" -eq 0 ] ; then
    duration="3600000"
    msval="msval"
    sed -i "s/$msval/${duration}/g" ../conf/gst_stress.conf
    date
    echo 'gst-1hr';../../../bin/rvs -c ../conf/gst_stress.conf -d 3; date
    sed -i "s/${duration}/$msval/g" ../conf/gst_stress.conf
else
    duration=$((3600000 * $1))
    echo $duration
    msval="msval"
    sed -i "s/$msval/${duration}/g" ../conf/gst_stress.conf
    date
    echo 'gst-'$1hr'(s)';../../../bin/rvs -c ../conf/gst_stress.conf -d 3; date
    sed -i "s/${duration}/$msval/g" ../conf/gst_stress.conf
fi
