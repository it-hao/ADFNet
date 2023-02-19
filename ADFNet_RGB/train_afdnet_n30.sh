CUDA_VISIBLE_DEVICES=0,1 python main.py --n_GPUs 2 --model adfnet --save adfnet_n30 --noiseL 30 --patch_size 128 --loss 1*MSE --lr 1e-4 --save_results --epochs 1000
