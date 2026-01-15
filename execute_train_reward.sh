# python -m scripts.train_reward --batch_size 1000 \
#                             --num_workers 40 \
#                             --reward_lr 1e-5 \
#                             --num_epochs 100 \
#                             --seed 777 \
#                             --dataset_dir og_dataset \
#                             --experiment 3B_discoverer_test_epoch100\
#                             --train_dir train_dir/skill_rewards

# python -m scripts.train_reward_ult --batch_size 1000 \
#                             --num_workers 48 \
#                             --reward_lr 1e-5 \
#                             --num_epochs 20 \
#                             --seed 777 \
#                             --dataset_dir og_dataset_copy \
#                             --experiment llama3.3_70B_foundry_food_eat_default \
#                             --train_dir train_dir/skill_rewards

python -m scripts.train_reward_ult --batch_size 1000 \
                            --num_workers 48 \
                            --reward_lr 1e-5 \
                            --num_epochs 100 \
                            --seed 777 \
                            --dataset_dir og_dataset_copy \
                            --experiment llama3.3_70B_foundry_scroll_read_default \
                            --train_dir train_dir/skill_rewards \
                            --exclude_label 4 \
                            --limit_pairs_fraction 0.5