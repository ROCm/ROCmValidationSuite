echo "=========================Starting RVS Suite========================="

echo "===========================gpup========================="
sudo ./gpup.new.sh 2>&1 | tee gpup.txt
echo "===========================gst========================="
sudo ./gst.new.sh  2>&1 | tee ggst.txt
echo "===========================iet========================="
sudo ./iet.new.sh  2>&1 | tee iet.txt
echo "===========================mix========================="
sudo ./mix.new.sh  2>&1 | tee mix.txt
echo "===========================pebb========================="
sudo ./pebb.new.sh  2>&1 | tee pebb.txt
echo "===========================peqt========================="
sudo ./peqt.new.sh  2>&1 | tee peqt.txt
echo "==========================pesm========================="
sudo  ./pesm.new.sh  2>&1 | tee pesm.txt
echo "===========================pqt========================="
sudo ./pqt.new.sh  2>&1 | tee pqt.txt
echo "===========================rand========================="
sudo ./rand.new.sh  2>&1 | tee rand.txt
echo "===========================rcqt========================="
sudo ./rcqt.new.sh  2>&1 | tee rcqt.txt
echo "===========================smqt========================="
sudo ./smqt.new.sh  2>&1 | tee smqt.txt
echo "===========================ttf========================="
sudo ./ttf.new.sh  2>&1 | tee ttf.txt

echo "=========================RVS Suite completed, check txt for output============"
