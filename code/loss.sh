#########################################################################
# File Name: loss.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Jul  9 01:31:12 2022
#########################################################################
#!/bin/bash

cat run.mgpu.launch.sh.log | python comp_avg_loss.py > avg_train_loss_per_epoch.log 2>&1 

tail -n 20 avg_train_loss_per_epoch.log | grep -v "Epoch"
