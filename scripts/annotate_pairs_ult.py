import argparse
import glob
import os
import random
import pickle
import tqdm

import numpy as np
import torch

from rlaif.data import get_dataset
from rlaif.annotators import RandomAnnotator, LanguageModelAnnotator
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

flags = parser.parse_args()

seed = flags.chunk_number + 1516
random.seed(seed)
np.random.seed(seed)

# Setup annotator
flags.logdir = "prompt_logs/" if flags.logdir is None else flags.logdir

if flags.custom_annotator_string is None:
    annotator_string = f"{flags.annotator_type}_{flags.goal_key}_{flags.prompt}"
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
                                       goal_key=flags.goal_key, 
                                       num_gpus=torch.cuda.device_count())
elif flags.annotator_type == 'llama3.2_3B':
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    annotator = LanguageModelAnnotator(seed=seed, batch_size=flags.batch_size, 
                                       debug=flags.debug,
                                       model_name=model_name, 
                                       annotator_string=annotator_string,
                                       logdir=flags.logdir, 
                                       prompt=flags.prompt,
                                       goal_key=flags.goal_key, 
                                       num_gpus=torch.cuda.device_count())
elif flags.annotator_type == 'random':
    annotator = RandomAnnotator(batch_size=flags.batch_size)
else:
    raise NotImplementedError

dataset, pairs_size = get_dataset(f"{flags.directory}/{flags.goal_key}.pkl")

# Load/create annotations
saving_path = os.path.join('preference')
os.makedirs(saving_path, exist_ok=True)
annotation_filename = os.path.join(saving_path, annotator_string + ".npy")

if flags.ignore_existing:
    annotation_array = np.ones((pairs_size,), dtype=np.int32)
    annotation_array[:] = AnnotationIdx.UNKOWN
else:
    annotation_array = np.load(annotation_filename)

# Main loop
chunk_size = int(pairs_size / flags.n_chunks)
assert chunk_size % flags.batch_size == 0

# Restrict the dataset to the portion that (1) is part of this chunk and (2) has the mask at False
low_idx = flags.chunk_number * pairs_size // flags.n_chunks
high_idx = (flags.chunk_number+1) * pairs_size // flags.n_chunks
indices = np.arange(low_idx, high_idx)[annotation_array[low_idx:high_idx] == AnnotationIdx.UNKOWN]

num_iterations = chunk_size // flags.batch_size
for i in tqdm.tqdm(range(num_iterations)):

    cur_batch = dataset[flags.batch_size * 2 * i: flags.batch_size * 2 * (i+1)]
    samples = []
    for j in range(0, len(cur_batch), 2):
        samples.append([cur_batch[j], cur_batch[j+1]])

    curr_idx = i * flags.batch_size
    end_idx = min((i+1) * flags.batch_size, chunk_size)

    inds = indices[list(range(curr_idx, end_idx))]

    annotation = annotator(batch=samples, logging_indices=inds, iteration=i)
    annotation_array[inds] = annotation

    if i % flags.flushing_freq == 0:
        np.save(annotation_filename, annotation_array)

# Final save
np.save(annotation_filename, annotation_array)
