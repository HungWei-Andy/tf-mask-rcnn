source .env/bin/activate
LOG="logs/test_resnet50_fpn_batch1.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
time python ./lib/test.py
