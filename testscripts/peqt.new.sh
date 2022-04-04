#!/bin/sh
date
../conf/deviceid.sh ../conf/peqt_single.conf
echo 'peqt';sudo ../../../bin/rvs -c ../conf/peqt_single.conf -d 3;date 
