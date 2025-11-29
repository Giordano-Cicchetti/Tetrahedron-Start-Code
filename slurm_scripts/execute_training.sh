#!/bin/sh
#SBATCH -A IscrC_VideoGen
#SBATCH -p boost_usr_prod
#SBATCH --time=23:59:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=MSRVTT-scratch

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_work/IscrC_VideoGen/Multimodal-Reranker
export WANDB_MODE=offline
conda activate triangle

output_dir=./output/gram/$config_name

save_dir=./output/gram/msrvtt_from_scratch_gram_TVA_cosine_reg
### VIDEO-RET


#retrieval-msrvtt
srun python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/gram/finetune_cfg/retrieval-msrvtt.json \
--output_dir $save_dir \
