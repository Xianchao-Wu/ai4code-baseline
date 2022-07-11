
python train.py \
	--md_max_len=64 \
	--total_max_len=512 \
	--num_code_cell=80 \
	--batch_size=24 \
	--epochs=20 \
	--n_workers=4 \
	--feature-file-path='./data_code80_clen50_val0.1' \
	--out-model-path='./outputs.1gpu.code80.clean50.v0.1' \
	--weights='./outputs/model_11_ktau0.8480169613968561.bin' \
	--device='cuda:3'

