#!/bin/bash
#SBATCH --job-name=z0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --partition=MIKO
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=h100:2
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xiaowen4@andrew.cmu.edu

hostname; pwd; date

source /hildafs/home/xzhangn/.bashrc
conda activate torch206

# export CUDA_LAUNCH_BLOCKING="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_root_dir="/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology"
code_root_dir="/hildafs/home/xzhangn/xzhangn/cosmo_sr/3-training/single_redshift_stylegan2"
in_dir="dmo-64"
tgt_dir="dmo-128"

train_dirs="set*/PART_009"

# Key improvements:
# 1. Progressive fade-in for first 50 epochs (half of 100 epochs)
# 2. Higher learning rates with better optimizer settings (betas=[0.0, 0.99])
# 3. Delayed adversarial training start (epoch 5 instead of 2)
# 4. Enabled pixel shuffle and normalization in misc_kwargs

srun python $code_root_dir/map2map/m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/disp.npy","$data_root_dir/$in_dir/$train_dirs/vel.npy" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/disp.npy","$data_root_dir/$tgt_dir/$train_dirs/vel.npy" \
    --train-style-pattern "$data_root_dir/$in_dir/$train_dirs/style.npy" \
    --in-norms cosmology.dis,cosmology.vel --tgt-norms cosmology.dis,cosmology.vel \
    --augment --aug-shift 28 \
    --crop 28 --crop-step 28 --pad 3 --scale-factor 2 --previous-scale-factor 0 --target-meshsize 512 \
    --model stylegan.G --adv-model stylegan.D --adv-criterion HingeLoss --cgan --callback-at . \
    --adv-start 5 --adv-wgan-gp-interval 16 \
    --lr 2e-4 --adv-lr 2e-4 --optimizer Adam --optimizer-args '{"betas": [0.0, 0.99]}' \
    --batch-size 1 --loader-workers 2 --load-state checkpoint.pt \
    --epochs 100 --progressive-fade-epochs 50 \
    --misc-kwargs '{"use_pixel_shuffle": true, "use_normalize": true}' \

date