
python train.py \
	--md_max_len=64 \
	--total_max_len=512 \
	--num_code_cell=30 \
	--batch_size=24 \
	--epochs=20 \
	--n_workers=2 \
	--feature-file-path='./data_code30_clen134_val0.01' \
	--out-model-path='./outputs.1gpu.code30.clen134.v0.01.L1' \
	--weights='./outputs/a100_model_0_ktau0.853742298611357.bin' \
	--device='cuda:6' \
	--loss='L1'

# use MSELoss()
# data_code30_clen134_val0.01
# a100_model_0_ktau0.853742298611357.bin
# --weights='./outputs/model_11_ktau0.8480169613968561.bin' \

