#!/bin/sh
date
../conf/deviceid.sh ../conf/iet_single.conf
echo 'iet';sudo ../../../bin/rvs -c ../conf/iet_single.conf -d 3 && date
