#!/bin/sh
date
../conf/deviceid.sh ../conf/pebb_single.conf
echo 'pebb';sudo ../../../bin/rvs -c ../conf/pebb_single.conf -d 3 ; date
