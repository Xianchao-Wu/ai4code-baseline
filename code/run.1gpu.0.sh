
python train.py \
	--md_max_len=64 \
	--total_max_len=512 \
	--num_code_cell=20 \
	--batch_size=96 \
	--epochs=20 \
	--n_workers=4 \
	--feature-file-path='./data' \
	--out-model-path='./outputs.1gpu.code20.clean200.v0.1' \
	--weights='./outputs/model_13_ktau0.8500557205835783.bin' \
	--device='cuda:1'

#--weights='./outputs/a100_model_1_ktau0.8542453426660417.cell40.bin' \
#--feature-file-path='./data_code20_clen200_val0.1' \

