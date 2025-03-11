python3 -m scripts.main --algo APPO --num_workers 24 --num_envs_per_worker 20 \
                       --batch_size 4096 --reward_scale 0.1 --obs_scale 255.0 \
                       --train_for_env_steps 5_000_000_000 --save_every_steps 6_000_000_000 \
                       --code_seed 43 --seed 2  --stats_avg 1000    \
                       --train_dir train_dir/skill_policy/ --experiment default \
                       --reward_dir "['train_dir/skill_rewards/llama3_discoverer_default',  \
                                      'train_dir/skill_rewards/llama3_descender_default',   \
                                      'train_dir/skill_rewards/llama3_ascender_default',    \
                                      'train_dir/skill_rewards/llama3_worshipper_default',  \
                                      'train_dir/skill_rewards/llama3_merchant_default']"   \
                         --evaluation True --eval_target goldenexit 