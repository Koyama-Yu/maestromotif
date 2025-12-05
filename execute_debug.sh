#!/bin/bash

GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    echo "使用するGPU番号を指定してください (例: bash run_appo.sh 0)"
    exit 1
fi

# GPU指定（実行時引数から）
export CUDA_VISIBLE_DEVICES=$GPU_ID

python -m scripts.main --algo APPO --num_workers 24 --num_envs_per_worker 20 \
                       --batch_size 4096 --reward_scale 0.1 --obs_scale 255.0 \
                       --train_for_env_steps 2_000_000 --save_every_steps 50_000_000 \
                       --keep_checkpoints 5 --stats_avg 1000 --seed 2 --code_seed -1 \
                       --reward_dir "[]" \
                       --extrinsic_reward 1.0 --llm_reward 0.0 \
                       --train_dir train_dir/skill_policy/ \
                       --experiment debug6 \
                       --ttyrec 1 \
                       --eval_target none