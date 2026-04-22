#!/bin/bash
#SBATCH --job-name=rl_loop
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=4

sleep 100000000000
