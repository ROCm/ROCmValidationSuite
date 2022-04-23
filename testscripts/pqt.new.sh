#!/bin/sh
date
../conf/deviceid.sh ../conf/pqt_single.conf
echo 'pqt';sudo ../../../bin/rvs -c ../conf/pqt_single.conf -d 3; date
