
import csv
import glob
import itertools
import json
import math
import os
import pickle
import sys
import tqdm
from collections import defaultdict, deque
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from rlaif.data import get_dataset
from rlaif.reward_model import create_reward_model, RunningMeanStd
from rlaif.save_helper import save
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.utils.arguments import (arg_parser, parse_args, 
    maybe_load_from_checkpoint)
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict, cfg_file, log, str2bool
from utils.preprocessing import DictAsAttributes

# Needs to be imported to register models and envs
import rl_baseline.monk_tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle

"""
python -m scripts.train_reward --experiment llama3_ascender_default --num_epochs 20 \
--batch_size 1000 --env nle_all_but_engrave --root_env NetHackScoreMonk-v1  \
--num_workers 40 --seed 0
"""

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recurse into the sublist
        else:
            flat_list.append(item)
    return flat_list

def train_reward(cfg):
    """
    This code will train the reward model, through binary cross entropy,
    to express the preferences over trajectories.
    """

    cfg = maybe_load_from_checkpoint(cfg)
    np.random.seed(cfg.seed)

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)
    dummy_env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))

    # Save experiment configuration
    with open(cfg_file(cfg), 'w') as json_file:
        json.dump(cfg, json_file, indent=2)

    ### Create the reward model
    reward_model = create_reward_model(
        cfg, dummy_env.observation_space, dummy_env.action_space
    )
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')

    # Possibly load a checkpoint
    checkpoints = LearnerWorker.get_checkpoints(
        os.path.join(cfg.train_dir, cfg.experiment, f'checkpoint_p0')
    )
    if len(checkpoints):
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        reward_model.load_state_dict(checkpoint_dict['model'])
        reward_model.add_mean_var(checkpoint_dict['reward_mean'].item(),
                                checkpoint_dict['reward_var'].item())
    reward_model.model_to_device(device)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(
        reward_model.parameters(), lr=cfg.reward_lr)
    loss_fn = lambda logprobs, target: \
        -(target * logprobs).sum() / logprobs.shape[0]

    # Define metrics
    train_metrics = {
        'epoch': [],
        'total_train_loss': [],
        'total_train_acc': [],
    }
    train_met = DictAsAttributes(train_metrics)
    val_metrics = {
        'iter': [],
        'total_val_loss': [],
        'validation_accs': [],
    }
    val_met = DictAsAttributes(val_metrics)

    # Get annotations and the dataset
    saving_path = os.path.join('preference')
    os.makedirs(saving_path, exist_ok=True)
    annotation_filename = os.path.join(saving_path, cfg.experiment + ".npy")
    annotations = np.load(annotation_filename)
    annotations = torch.from_numpy(annotations)
    agent = cfg.experiment.split('_')[1]
    all_samples , _ = get_dataset(f"{cfg.dataset_dir}/{agent}.pkl")
    all_msgs_input, pairs_inputs = get_all_messages(cfg, all_samples, agent)

    if len(pairs_inputs['message']) != len(annotations):
        print("Shapes are wrong!")
        input()
        annotations = annotations[:len(pairs_inputs['message'])]
    
    # Training
    tot_num_iter = 0
    dataset_size = len(annotations)
    msg_shape = 256
    diff_shape = 5

    assert dataset_size % cfg.batch_size == 0
    num_iterations = int(dataset_size / cfg.batch_size)
    for epoch in range(cfg.num_epochs):
        train_loss = 0.
        train_acc = 0.
        for i in range(num_iterations):

            if tot_num_iter % num_iterations == 0:
                full_reward_rms, all_msgs_rewards = get_rms(cfg, reward_model,
                                                            all_msgs_input, 
                                                            device)

                print(f'Full Reward mean: {full_reward_rms.mean[0]:.3f} '
                      f'Full Reward variance: {full_reward_rms.var[0]:.3f}')
                save(cfg, tot_num_iter, reward_model, optimizer, 
                     train_met._data_dict, val_met._data_dict, 
                     full_reward_rms, all_msgs_rewards, all_msgs_input)

            indices = list(range(i * cfg.batch_size, (i+1) * cfg.batch_size))

            mb = {}
            mb['message'] = pairs_inputs['message'][indices].reshape(-1, msg_shape)
            mb['diffstats'] = pairs_inputs['diffstats'][indices].reshape(-1, diff_shape)
            cur_annotations = annotations[indices]

            result = reward_model.forward_pairs(mb)
            rewards = result.rewards # sequence length x BS x 2
            rewards = rewards.mean(axis=0)

            labels = cur_annotations.to(device).type(torch.float) # BS
            labels[torch.where(labels > 1)[0]] = 0.5
            soft_labels = torch.zeros(len(rewards), 2, device=device)

            soft_labels[:, 1] = labels
            soft_labels[:, 0] = 1. - labels

            predicted_log_probs = nn.functional.log_softmax(rewards, dim=1)
            loss = loss_fn(predicted_log_probs, soft_labels)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()  # Perform back-propagation
            optimizer.step()  # Update the weights

            reward_argmax = np.argmax(rewards.detach().cpu().numpy(), axis=1)
            labels = labels.cpu().numpy()
            train_acc += np.mean(
                reward_argmax[labels != 0.5] == labels[labels != 0.5])
            cur_acc = train_acc / (i+1)

            tot_num_iter += 1

        train_met.total_train_loss.append(train_loss / num_iterations)
        train_met.total_train_acc.append(train_acc / num_iterations)
        train_met.epoch.append(epoch)

        print(f"==== Epoch {epoch} ====\n"
              f"Iteration {tot_num_iter} "
              f"Training accuracy: {train_met.total_train_acc[-1]:.4f}\n"
              f"Iteration {tot_num_iter} "
              f"Train loss: {train_met.total_train_loss[-1]:.4f}")

    print(f'Saving final model...')

    full_reward_rms, all_msgs_rewards = get_rms(cfg, reward_model,
                                                all_msgs_input, 
                                                device)

    save(cfg, tot_num_iter, reward_model, optimizer, train_met._data_dict, 
         val_met._data_dict, full_reward_rms, all_msgs_rewards, all_msgs_input)


def get_all_messages(cfg, sampled_data, agent):
    all_msgs_input = {'all_diffstats': [],
                      'all_message': [],
                      'all_message_counts': [],
                      'all_message_strs': []}

    def get_stat_str(combo):
        dlvl = [' | Dlvl: none', ' | Dlvl: down', ' | Dlvl: up']
        xp = [' | Xp: none', ' | Xp: up', ' | Xp: down']
        hp = [' | HP: none', ' | HP: up', ' | HP: down']
        gold = [' | Gold: none', ' | Gold: up', ' | Gold: down']
        hunger = [' | Hunger: none', ' | Hunger: more', ' | Hunger: less']
        stats_str = [dlvl, gold, hp, xp, hunger]

        string = ''
        for val, strs in zip(combo, stats_str):
            if val == -1:
                val = 2
            string += strs[val]
        return string

    save_data = []
    for sample in sampled_data:
        all_msgs_input['all_diffstats'].append(sample['diffstats'])
        all_msgs_input['all_message'].append(sample['messages'][1])
        all_msgs_input['all_message_strs'].append(sample['msg_strs'][1] + get_stat_str(sample['diffstats'].astype(int)))
        all_msgs_input['all_message_counts'].append(1)

    all_msgs_input['all_diffstats'] = torch.from_numpy(np.array(all_msgs_input['all_diffstats'], dtype=np.float32))
    all_msgs_input['all_message'] = torch.from_numpy(np.array(all_msgs_input['all_message'], dtype=np.float32))

    pairs_inputs = {'message': all_msgs_input['all_message'].reshape(-1, 2, 256),
                    'diffstats': all_msgs_input['all_diffstats'].reshape(-1, 2, 5)}

    def unique_indices_only(strings):
        index_map = {}
        for index, value in enumerate(strings):
            if value not in index_map:
                index_map[value] = index
        return list(index_map.values())

    inds = unique_indices_only(all_msgs_input['all_message_strs'])
    msgs_input = {}
    for key, value in all_msgs_input.items():
        if 'strs' in key or 'counts' in key:
            data = list(np.array(value)[inds])
        else:
            data = value[inds]
        if 'all_' not in key:
            continue
        msgs_input[key.split('all_')[1]] = data

    return all_msgs_input | msgs_input, pairs_inputs


def get_rms(cfg, reward_model, all_msgs_input, device):
    print("Calculating RMS of the reward function...")
    reward_rms = RunningMeanStd(device)
    all_msgs_rewards = []
    dataset_size = len(all_msgs_input['message'])
    with torch.no_grad():
        num_iterations = int(dataset_size / cfg.batch_size) + 1
        for i in range(num_iterations):
            indices = range(i * cfg.batch_size, min((i+1) * cfg.batch_size, dataset_size) )

            mb = {}
            mb['message'] = all_msgs_input['message'][indices]
            mb['diffstats'] = all_msgs_input['diffstats'][indices]
            result = reward_model(mb)

            rewards = result.rewards # sequence length x BS x 2
            rewards = rewards.reshape(-1, 1)
            all_msgs_rewards.append(rewards)
            reward_rms.update(rewards)

    all_msgs_rewards = torch.vstack(all_msgs_rewards)
    return reward_rms, all_msgs_rewards.flatten()


def add_extra_params(parser):
    """
    Specify any additional command line arguments.
    """
    p = parser
    p.add_argument("--reward_lr", default=1e-5, type=float, 
                   help="Reward model learning rate")
    p.add_argument("--num_epochs", default=5, type=int, 
                   help="The number of epochs to train the reward model.")
    p.add_argument("--dataset_dir", default='og_dataset', type=str, 
                   help="Directory from which we load the dataset.")
    p.add_argument("--debug", default=False, type=str2bool, 
                   help="To debug or not the reward model")


def parse_all_args(argv=None, evaluation=True):
    parser = arg_parser(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Evaluation entry point."""
    cfg = parse_all_args()
    train_reward(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())
