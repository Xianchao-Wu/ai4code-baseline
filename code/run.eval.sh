#########################################################################
# File Name: run.eval.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Jul  8 23:12:36 2022
#########################################################################
#!/bin/bash

python evaluate.py \
	--batch_size=16 \
	--device='cuda:0'

#for afile in `ls outputs.mgpu/*`
#do
#	echo $afile
#	python evaluate.py \
#		--weights=$afile \
#		--batch_size=4 \
#		--device='cuda:0' 
#	echo "----"
#done
