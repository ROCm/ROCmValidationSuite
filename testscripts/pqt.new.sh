#!/bin/sh
date
../conf/deviceid.sh ../conf/pbqt_single.conf
echo 'pbqt';sudo ../../../bin/rvs -c ../conf/pbqt_single.conf -d 3; date
