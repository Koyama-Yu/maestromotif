python -m scripts.train_reward --batch_size 1000 \
                            --num_workers 40 \
                            --reward_lr 1e-5 \
                            --num_epochs 100 \
                            --seed 777 \
                            --dataset_dir og_dataset \
                            --experiment 3B_discoverer_test_epoch100\
                            --train_dir train_dir/skill_rewards