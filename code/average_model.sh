#########################################################################
# File Name: average_model.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Jul 11 07:25:29 2022
#########################################################################
#!/bin/bash

#nbest=5
nbest=10

python average_model.py \
	--dst_model="./outputs/average_${nbest}_best.bin" \
	--src_path='./outputs/' \
	--val_best \
	--num=$nbest 
