
python train.py \
	--md_max_len=64 \
	--total_max_len=512 \
	--num_code_cell=50 \
	--batch_size=24 \
	--epochs=20 \
	--n_workers=2 \
	--feature-file-path='./data_code50_clen80_val0.01' \
	--out-model-path='./outputs.1gpu.code50.clen80.v0.01.L1' \
	--weights='./outputs/model_11_ktau0.8480169613968561.bin' \
	--device='cuda:7' \
	--loss='L1'

