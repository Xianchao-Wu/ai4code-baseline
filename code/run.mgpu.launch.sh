#########################################################################
# File Name: run.launch.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Jul  7 11:15:22 2022
#########################################################################
#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch \
	--nproc_per_node=6 --use_env train_mgpu.py \
	--n_workers=6
