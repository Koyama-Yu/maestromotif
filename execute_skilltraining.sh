python -m scripts.main --algo APPO \
                       --num_workers 24 \
                       --num_envs_per_worker 20 \
                       --batch_size 4096 \
                       --reward_scale 0.1 \
                       --obs_scale 255.0 \
                       --train_for_env_steps 4_000_000_000 \
                       --save_every_steps 50_000_000 \
                       --keep_checkpoints 5 \
                       --stats_avg 1000 \
                       --seed 2 \
                       --code_seed 42  \
                       --reward_dir "['train_dir/skill_rewards/llama3_discoverer_default',
                                      'train_dir/skill_rewards/llama3_descender_default',
                                      'train_dir/skill_rewards/llama3_ascender_default',
                                      'train_dir/skill_rewards/llama3_worshipper_default',
                                      'train_dir/skill_rewards/llama3_merchant_default']" \
                       --extrinsic_reward 1.0 \
                       --llm_reward 0.1 \
                       --train_dir train_dir/skill_policy/ \
                       --experiment test_extrinsic