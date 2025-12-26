from abc import ABC, abstractmethod
import itertools
from typing import Dict, List, Callable, Optional, Tuple, Sequence

import numpy as np
import torchvision

from rlaif.annotators_transforms import BlstatsTransform, MessageTransform
from rlaif import prompts as prompts_default
from rlaif import prompts_foritem as prompts_items
from rlaif.llms import LocalLanguageModel, AnnotationIdx


class Annotator(ABC):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int]) -> np.array:
        """General method which takes two sequences and returns whether the second element
        is better than the first one, for each batch element,
        together with a mask of the valid/invalid elements.

        Args:
            batch: Dictionary of arrays containing the data for the two sequences (bs, 2, subepisode_length, dims)
            logging_indices: a list of indices for logging info about computation for each element
        Return:
            annotations: int array of shape (bs,) where each element is out of (first, second, tie, invalid)
        """
        pass

    @property
    @abstractmethod
    def data_keys(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def info_keys(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def transform(self) -> Optional[Callable]:
        pass


class LanguageModelAnnotator(Annotator):
    """Annotator that annotates based on the output of a language model."""
    def __init__(self, seed: int, batch_size: int, debug: int, model_name: str, annotator_string: str,
                 num_gpus: int = 8, logdir: Optional[str] = None,
                 prompt: str = 'original',
                 goal_key: str = '') -> None:

        self.blstats_keys = [
           'NLE_BL_DEPTH', 'NLE_BL_GOLD', 'NLE_BL_HP',
           'NLE_BL_HPMAX', 'NLE_BL_XP', 'NLE_BL_HUNGER'
        ]
        prompts_module = prompts_items if goal_key in prompts_items.goal_strings else prompts_default

        if debug:
            self.llm = None
        else:
            self.llm = LocalLanguageModel(seed=seed, system_prompt=prompts_module.system_prompts[prompt],
                                        answer_regex=prompts_module.regexes[prompt],
                                        retry_prompt=prompts_module.retry_prompts[prompt],
                                        model_name=model_name, num_gpus=num_gpus,
                                        logdir=logdir, annotator_string=annotator_string)

        self.prompt_template = prompts_module.prompt_templates[prompt]
        self.goal_key = goal_key
        super().__init__(batch_size)

    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None, iteration: int = 0) -> np.ndarray:
        messages = list(map(lambda x: [x[0]['llm_strs'], x[1]['llm_strs']], batch))
        prompts, preserved_indices = self.prepare_prompts(messages)
        print('Sample prompt:')
        print(prompts[0])

        results = self.llm.generate(prompts,
                                    np.array(logging_indices)[preserved_indices] if logging_indices is not None else None,
                                    iteration)

        recomposed_results = np.full(len(messages), AnnotationIdx.TIE)
        recomposed_results[preserved_indices] = results
        return recomposed_results

    def prepare_prompts(self, batched_messages: List[List[str]],) -> Tuple[List[str], List[int]]:

        preserved_indices = []
        prompts = []
        for prompt_idx, (seq_1, seq_2) in enumerate(batched_messages):
            seq_1 = "\n".join(seq_1)
            seq_2 = "\n".join(seq_2)
            preserved_indices.append(prompt_idx)
            prompts.append(self.prompt_template.format(goal_strings[self.goal_key], seq_1, seq_2))
        return prompts, preserved_indices

    @property
    def data_keys(self) -> List[str]:
        needed_keys = []
        if self.use_messages:
            needed_keys.append('message')
        if self.use_blstats:
            needed_keys.append('blstats')
        return needed_keys

    @property
    def info_keys(self) -> Optional[List[str]]:
        return None

    @property
    def transform(self):
        transforms = []
        if self.use_messages:
            transforms.append(MessageTransform())
        if self.use_blstats:
            transforms.append(BlstatsTransform(self.blstats_keys))
        return torchvision.transforms.Compose(transforms)
    

#! 追記
# class OllamaModelAnnotator(Annotator):
#     """Annotator that annotates based on the output of a local Ollama model."""
#     def __init__(self, seed: int, batch_size: int, debug: int, model_name: str, annotator_string: str,
#                  num_gpus: int = 8, logdir: Optional[str] = None,
#                  prompt: str = 'original',
#                  goal_key: str = '') -> None:

#         self.blstats_keys = [
#            'NLE_BL_DEPTH', 'NLE_BL_GOLD', 'NLE_BL_HP',
#            'NLE_BL_HPMAX', 'NLE_BL_XP', 'NLE_BL_HUNGER'
#         ]
#         if debug:
#             self.llm = None
#         else:
#             self.llm = OllamaLanguageModel(seed=seed, system_prompt=system_prompts[prompt],
#                                            answer_regex=regexes[prompt],
#                                            retry_prompt=retry_prompts[prompt],
#                                            model_name=model_name, num_gpus=num_gpus,
#                                            logdir=logdir, annotator_string=annotator_string)

#         self.prompt_template = prompt_templates[prompt]
#         self.goal_key = goal_key
#         super().__init__(batch_size)

#     def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None, iteration: int = 0) -> np.ndarray:
#         messages = list(map(lambda x: [x[0]['llm_strs'], x[1]['llm_strs']], batch))
#         prompts, preserved_indices = self.prepare_prompts(messages)
#         print('Sample prompt:')
#         print(prompts[0])

#         results = self.llm.generate(prompts,
#                                     np.array(logging_indices)[preserved_indices] if logging_indices is not None else None,
#                                     iteration)

#         recomposed_results = np.full(len(messages), AnnotationIdx.TIE)
#         recomposed_results[preserved_indices] = results
#         return recomposed_results

#     def prepare_prompts(self, batched_messages: List[List[str]],) -> Tuple[List[str], List[int]]:

#         preserved_indices = []
#         prompts = []
#         for prompt_idx, (seq_1, seq_2) in enumerate(batched_messages):
#             seq_1 = "\n".join(seq_1)
#             seq_2 = "\n".join(seq_2)
#             preserved_indices.append(prompt_idx)
#             prompts.append(self.prompt_template.format(goal_strings[self.goal_key], seq_1, seq_2))
#         return prompts, preserved_indices

#     @property
#     def data_keys(self) -> List[str]:
#         needed_keys = []
#         if self.use_messages:
#             needed_keys.append('message')
#         if self.use_blstats:
#             needed_keys.append('blstats')
#         return needed_keys

#     @property
#     def info_keys(self) -> Optional[List[str]]:
#         return None

#     @property
#     def transform(self):
#         transforms = []
#         if self.use_messages:
#             transforms.append(MessageTransform())
#         if self.use_blstats:
#             transforms.append(BlstatsTransform(self.blstats_keys))
#         return torchvision.transforms.Compose(transforms)


class RandomAnnotator(Annotator):
    """Annotator that annotates randomly."""
    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None, iteration: int = 0) -> np.ndarray:
        return np.random.choice([0, 1, 2], size=(self.batch_size,))

    @property
    def data_keys(self) -> Optional[List[str]]:
        return None

    @property
    def info_keys(self) -> Optional[List[str]]:
        return None

    @property
    def transform(self):
        return None
