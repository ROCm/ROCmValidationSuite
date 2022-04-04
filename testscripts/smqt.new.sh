#!/bin/sh
date
echo 'smqt_1';sudo ../../../bin/rvs -c ../conf/smqt_1.conf -d 3; date
echo 'smqt_2';sudo ../../../bin/rvs -c ../conf/smqt_2.conf -d 3; date
echo 'smqt_3';sudo ../../../bin/rvs -c ../conf/smqt_3.conf -d 3; date

