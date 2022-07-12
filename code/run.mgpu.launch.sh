#########################################################################
# File Name: run.launch.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Jul  7 11:15:22 2022
#########################################################################
#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch \
	--nproc_per_node=6 --use_env train_mgpu.py \
	--n_workers=6 \
	--md_max_len=64 \
	--total_max_len=512 \
	--num_code_cell=40 \
	--feature-file-path='./data_code40_clen100_val0.1' \
	--out-model-path='./outputs.multigpu.code40.clean100.v0.1' \
	--weights="./outputs/a100_model_1_ktau0.8542453426660417.cell40.bin" \
	--batch_size=96 \
	--epochs=50
