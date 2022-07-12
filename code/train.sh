#########################################################################
# File Name: train.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Jul  7 05:51:42 2022
#########################################################################
#!/bin/bash

#python -m ipdb train.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py \
#python -m ipdb train.py \
python train.py \
	--md_max_len=64 \
	--total_max_len=512 \
	--batch_size=16 \
	--accumulation_steps=4 \
	--epochs=50 \
	--n_workers=2
