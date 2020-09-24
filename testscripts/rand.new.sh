#!/bin/sh
date
./conf/deviceid.sh conf/rand_single.conf
echo 'rand';sudo ./rvs -c conf/rand_single.conf -d 3; date
