#!/bin/sh
date
../conf/deviceid.sh ../conf/gst_single.conf
echo 'gst';sudo ../../../bin/rvs -c ../conf/gst_single.conf -d 3; date 
