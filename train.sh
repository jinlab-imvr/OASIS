CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--master_port 12345 \
--nproc_per_node=4 \
oasis/train.py \
exp_id=main_small \
model=small \
data=base