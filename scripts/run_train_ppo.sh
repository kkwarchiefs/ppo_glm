export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node 2 --master_port=12345 train_ppo.py
accelerate launch  train_ppo.py
