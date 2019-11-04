mkdir -p $HOME/rvs/log-$1
 logs=log-$1
 password=amd 

echo "===========================gpup========================="
echo $password | sudo -S ./gpup.new.sh 2>&1 | tee gpup.txt
echo "===========================gst========================="
echo $password | sudo -S ./gst.new.sh  2>&1 | tee ggst.txt
echo "===========================iet========================="
echo $password | sudo -S ./iet.new.sh  2>&1 | tee iet.txt
echo "===========================mix========================="
echo $password | sudo -S ./mix.new.sh  2>&1 | tee mix.txt
echo "===========================pebb========================="
echo $password | sudo -S ./pebb.new.sh  2>&1 | tee pebb.txt
echo "===========================peqt========================="
echo $password | sudo -S ./peqt.new.sh  2>&1 | tee peqt.txt
echo "==========================pesm========================="
echo $password | sudo -S ./pesm.new.sh  2>&1 | tee pesm.txt
echo "===========================pqt========================="
echo $password | sudo -S ./pqt.new.sh  2>&1 | tee pqt.txt
echo "===========================rand========================="
echo $password | sudo -S ./rand.new.sh  2>&1 | tee rand.txt
echo "===========================rcqt========================="
echo $password | sudo -S ./rcqt.new.sh  2>&1 | tee rcqt.txt
echo "===========================smqt========================="
echo $password | sudo -S ./smqt.new.sh  2>&1 | tee smqt.txt
echo "===========================ttf========================="
echo $password | sudo -S ./ttf.new.sh  2>&1 | tee ttf.txt

mv *.txt $logs

