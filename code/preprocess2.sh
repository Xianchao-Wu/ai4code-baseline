#########################################################################
# File Name: preprocess.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Jul  9 12:54:47 2022
#########################################################################
#!/bin/bash

#python -m ipdb preprocess.py
#python preprocess.py --json_file_num=10

keep_code_cell_num=60
keep_code_len=65
valid_ratio=0.1

out_data_dir="./data_code${keep_code_cell_num}_clen${keep_code_len}_val${valid_ratio}"
echo ${out_data_dir}

python preprocess.py \
	--json_file_num=-1 \
	--out_data_dir=${out_data_dir} \
	--keep_code_cell_num=${keep_code_cell_num} \
	--keep_code_len=${keep_code_len} \
	--valid_ratio=${valid_ratio} 
