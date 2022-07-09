#########################################################################
# File Name: run.launch.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Jul  7 11:15:22 2022
#########################################################################
#!/bin/bash

python train.py \
	--n_workers=8 \
	--weights="./outputs/model_3_ktau0.8381995542759781.bin" \
	--batch_size=64
