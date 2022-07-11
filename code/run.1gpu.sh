
python train.py \
	--md_max_len=64 \
	--total_max_len=512 \
	--num_code_cell=40 \
	--batch_size=24 \
	--epochs=20 \
	--n_workers=4 \
	--feature-file-path='./data_code40_clen100_val0.1' \
	--out-model-path='./outputs.1gpu.code40.clean100.v0.1' \
	--weights='./outputs/model_11_ktau0.8480169613968561.bin' \
	--device='cuda:1'

