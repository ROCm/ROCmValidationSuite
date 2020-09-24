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
echo "===========================pqt========================="
sudo ./pqt.new.sh  2>&1 | tee pqt.txt
echo "===========================memory========================="
sudo ./rvs-mem.sh  2>&1 | tee memory.txt

echo "=========================RVS Suite completed, check txt for output============"
