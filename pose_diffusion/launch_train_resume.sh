#!/bin/bash
#SBATCH -p learn                   # partition (queue)
#SBATCH -N 1                       # number of nodes
#SBATCH --gres=gpu:8               # number of GPUs
#SBATCH --cpus-per-gpu=12          # CPU cores per GPU
#SBATCH --account=repligen         # account
#SBATCH --qos=low                  # QoS
#SBATCH --time=7-00:00:00          # time (D-HH:MM:SS)
#SBATCH --mem=0
#SBATCH --job-name="resume_train" # Descriptive job name


# Command to run your training
accelerate launch --num_processes=8 --multi_gpu --num_machines=1 train.py exp_dir=vggsfm_camera_resume train.len_train=8192 train.resume_ckpt=modified_checkpoint.bin





