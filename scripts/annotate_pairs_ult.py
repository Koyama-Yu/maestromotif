import argparse
import glob
import os
import random
import pickle
import sys
import tqdm

import numpy as np
import torch

from rlaif.data import get_dataset
from rlaif.annotators import RandomAnnotator, LanguageModelAnnotator, FoundryLanguageModelAnnotator
from rlaif.llms import AnnotationIdx


parser = argparse.ArgumentParser()
parser.add_argument('--annotator_type', type=str, default='llama3',
                     help="Type of annotator to use.")
parser.add_argument('--directory', type=str, default='og_dataset',
                     help="Directory of the dataset")
parser.add_argument('--custom_annotator_string', type=str, default=None,
                    help="Custom tag to be used for the annotation, overriding the default one.")

# Parameters used only for the llama annotator
parser.add_argument('--prompt', type=str, default='default',
                    help="The prompt to use.")
parser.add_argument('--goal_key', type=str, default='discoverer',
                    help="Key for the behavior-specification string to be added to the prompt.")
parser.add_argument('--logdir', type=str, default=None,
                    help="Name of the directory to log the conversations of the LLM.")
parser.add_argument('--api_key', type=str, default=None,
                    help="API key for Foundry-hosted models (required for Foundry annotators).")
parser.add_argument('--endpoint', type=str, default=None,
                    help="Foundry endpoint URL (required for Foundry annotators).")
parser.add_argument('--deployment_name', type=str, default='Llama-3.3-70B-Instruct',
                    help="Foundry deployment name for the model.")

# "System" parameters
parser.add_argument('--batch_size', type=int, default=2000,
                    help="Number of prompts that will be processed continuously.")
parser.add_argument('--n_chunks', type=int, default=1,
                    help="Number of chunks to split the dataset into.")
parser.add_argument('--chunk_number', type=int, default=0,
                    help="Chunk number that this instance of the script will process.")
parser.add_argument('--flushing_freq', type=int, default=5,
                    help='Number of batches after which the annotations will be flushed to disk.')
parser.add_argument('--debug', type=int, default=0,
                    help='To debug or not the code.')
parser.add_argument('--ignore_existing', type=int, default=0,
                    help='To ignore_existing some experiments.')
parser.add_argument('--unknown_limit', type=int, default=10,
                    help='Exit early if unknown annotations reach this count.')
parser.add_argument('--max_batches', type=int, default=0,
                    help='Limit the number of processed batches (0 means no limit).')

flags = parser.parse_args()

ITEM_ACTIONS = {
    'food': ['food_eat', 'food_drop'],
    'potion': ['potion_quaff', 'potion_drop'],
    'scroll': ['scroll_read', 'scroll_drop'],
    'wand': ['wand_zap', 'wand_drop'],
    'weapon': ['weapon_wield', 'weapon_throw', 'weapon_drop'],
    'armor': ['armor_wear', 'armor_take_off', 'armor_drop'],
}

seed = flags.chunk_number + 1516
random.seed(seed)
np.random.seed(seed)

# Setup annotator
flags.logdir = "prompt_logs/" if flags.logdir is None else flags.logdir

dataset_goal_key = flags.goal_key
action_goal_keys = ITEM_ACTIONS.get(flags.goal_key, [flags.goal_key])
dataset, pairs_size = get_dataset(f"{flags.directory}/{dataset_goal_key}.pkl")

# Load/create annotations
saving_path = os.path.join('preference')
os.makedirs(saving_path, exist_ok=True)

# Main loop
chunk_size = int(pairs_size / flags.n_chunks)
assert chunk_size % flags.batch_size == 0

for action_goal_key in action_goal_keys:
    if flags.custom_annotator_string is None:
        annotator_string = f"{flags.annotator_type}_{action_goal_key}_{flags.prompt}"
    else:
        if len(action_goal_keys) > 1:
            annotator_string = f"{flags.custom_annotator_string}_{action_goal_key}"
        else:
            annotator_string = flags.custom_annotator_string

    # Load the annotator
    if flags.annotator_type == 'llama3.1_70B':
        model_name = 'meta-llama/Llama-3.1-70B-Instruct'
        annotator = LanguageModelAnnotator(seed=seed, batch_size=flags.batch_size, 
                                           debug=flags.debug,
                                           model_name=model_name, 
                                           annotator_string=annotator_string,
                                           logdir=flags.logdir, 
                                           prompt=flags.prompt,
                                           goal_key=action_goal_key, 
                                           num_gpus=torch.cuda.device_count())
    elif flags.annotator_type == 'llama3.2_3B':
        model_name = 'meta-llama/Llama-3.2-3B-Instruct'
        annotator = LanguageModelAnnotator(seed=seed, batch_size=flags.batch_size, 
                                           debug=flags.debug,
                                           model_name=model_name, 
                                           annotator_string=annotator_string,
                                           logdir=flags.logdir, 
                                           prompt=flags.prompt,
                                           goal_key=action_goal_key, 
                                           num_gpus=torch.cuda.device_count())
    elif flags.annotator_type == 'llama3.3_70B_foundry':
        if not flags.api_key:
            raise ValueError("Missing --api_key for Foundry annotator.")
        if not flags.endpoint:
            raise ValueError("Missing --endpoint for Foundry annotator.")
        annotator = FoundryLanguageModelAnnotator(seed=seed, batch_size=flags.batch_size,
                                                  debug=flags.debug,
                                                  annotator_string=annotator_string,
                                                  endpoint=flags.endpoint,
                                                  api_key=flags.api_key,
                                                  deployment_name=flags.deployment_name,
                                                  logdir=flags.logdir,
                                                  prompt=flags.prompt,
                                                  goal_key=action_goal_key)
    elif flags.annotator_type == 'random':
        annotator = RandomAnnotator(batch_size=flags.batch_size)
    else:
        raise NotImplementedError

    annotation_filename = os.path.join(saving_path, annotator_string + ".npy")

    if flags.ignore_existing:
        annotation_array = np.ones((pairs_size,), dtype=np.int32)
        annotation_array[:] = AnnotationIdx.UNKOWN
    else:
        annotation_array = np.load(annotation_filename)

    # Restrict the dataset to the portion that (1) is part of this chunk and (2) is still unknown.
    low_idx = flags.chunk_number * pairs_size // flags.n_chunks
    high_idx = (flags.chunk_number + 1) * pairs_size // flags.n_chunks
    indices = np.arange(low_idx, high_idx)[annotation_array[low_idx:high_idx] == AnnotationIdx.UNKOWN]

    if len(indices) == 0:
        print("No unknown annotations in this chunk. Skipping.")
        continue

    num_iterations = (len(indices) + flags.batch_size - 1) // flags.batch_size
    if flags.max_batches > 0:
        num_iterations = min(num_iterations, flags.max_batches)
    unknown_count = 0
    for i in tqdm.tqdm(range(num_iterations)):
        curr_idx = i * flags.batch_size
        end_idx = min((i + 1) * flags.batch_size, len(indices))
        inds = indices[curr_idx:end_idx]

        samples = []
        for pair_idx in inds:
            dataset_idx = pair_idx * 2
            samples.append([dataset[dataset_idx], dataset[dataset_idx + 1]])

        try:
            annotation = annotator(batch=samples, logging_indices=inds, iteration=i)
        except Exception as exc:
            np.save(annotation_filename, annotation_array)
            print(f"Annotator failed at batch {i}: {exc}")
            print("Saved progress; please retry to resume from the first unknown index.")
            sys.exit(1)

        annotation_array[inds] = annotation
        unknown_count += int(np.count_nonzero(annotation == AnnotationIdx.UNKOWN))
        if unknown_count >= flags.unknown_limit:
            np.save(annotation_filename, annotation_array)
            print(f"Reached unknown_limit={flags.unknown_limit}. Exiting early.")
            sys.exit(0)

        if i % flags.flushing_freq == 0:
            np.save(annotation_filename, annotation_array)

    # Final save
    np.save(annotation_filename, annotation_array)
    unknown_total = int(np.count_nonzero(annotation_array == AnnotationIdx.UNKOWN))
    print(f"Total unknown annotations: {unknown_total}")
