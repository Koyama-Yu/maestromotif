import re
import time
from fastchat.model.model_adapter import get_conversation_template
from vllm import LLM, SamplingParams
import os
from typing import List, Optional, Sequence
import numpy as np
import ollama
from openai import (
    OpenAI,
    BadRequestError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class AnnotationIdx:
    FIRST = 0
    SECOND = 1
    TIE = 2
    UNKOWN = 3
    CONTENT_FILTER = 4


class LocalLanguageModel:
    def __init__(
        self,
        seed: int,
        system_prompt: str,
        answer_regex: str,
        retry_prompt: str,
        model_name: str = 'meta-llama/Llama-3.1-70B-Instruct',
        num_gpus: int = 8,
        logdir: Optional[str] = None,
        annotator_string: Optional[str] = 'all_convs',
    ) -> None:

        self.model_name = model_name
        self.answer_regex = answer_regex
        self.retry_prompt = retry_prompt
        self.llm = LLM(model=model_name, tensor_parallel_size=num_gpus,
                       dtype='float16', seed=seed, max_model_len=(32768 // 8))
        self.all_convs = ''
        self.system_prompt = system_prompt
        self.logdir = logdir
        if self.logdir is not None:
            os.makedirs(self.logdir, exist_ok=True)
        self.annotator_string = annotator_string

    def generate(self, messages: List[str], logging_indices: Sequence[int] = None, iteration: int = 0) -> List[int]:
        assert len(messages) == len(logging_indices)
        prompts = []
        convs = []

        for message in messages:
            conv = get_conversation_template(self.model_name) # get_conversation_template は fastchat.model.model_adapter にある
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            conv.system = self.system_prompt
            prompt = conv.get_prompt()
            prompts.append(prompt)
            convs.append(conv)

        sampling_params = SamplingParams(top_k=50, max_tokens=4096,
                                         temperature=0.1, top_p=0.95,
                                         stop=conv.stop_str)
        outputs = self.llm.generate(prompts, sampling_params)

        # Parse all the outputs
        cleaned_outputs = np.full(len(messages), AnnotationIdx.UNKOWN)
        indexes_to_retry = []
        prompts_to_retry = []
        for i, output in enumerate(outputs):
            text_answer = output.outputs[0].text
            result = re.search(self.answer_regex, text_answer)
            conv = convs[i]
            conv.append_message('', text_answer)
            if result:
                try:
                    best_sequence = int(result.group(1))
                    if best_sequence == 1:
                        best_sequence = AnnotationIdx.FIRST
                    elif best_sequence == 2:
                        best_sequence = AnnotationIdx.SECOND
                except ValueError:
                    best_sequence = AnnotationIdx.TIE
                cleaned_outputs[i] = best_sequence
            else:
                # Ask the model again
                conv.append_message(conv.roles[0], self.retry_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts_to_retry.append(prompt)
                indexes_to_retry.append(i)

        # Retry the prompts that were not good
        print("Retrying prompts")
        second_batch = self.llm.generate(prompts_to_retry, sampling_params)
        for i, output in zip(indexes_to_retry, second_batch):
            text_answer = output.outputs[0].text
            convs[i].append_message('', text_answer)
            result = re.search(self.answer_regex, text_answer)
            if result:
                try:
                    best_sequence = int(result.group(1))
                    if best_sequence == 1:
                        best_sequence = AnnotationIdx.FIRST
                    elif best_sequence == 2:
                        best_sequence = AnnotationIdx.SECOND
                except ValueError:
                    best_sequence = AnnotationIdx.TIE
                cleaned_outputs[i] = best_sequence

        # Log the conversations
        if self.logdir is not None and logging_indices is not None:
            log_path = os.path.join(self.logdir, f"{self.annotator_string}.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                for conv, idx in zip(convs, logging_indices):
                    text_conv = conv.get_prompt()
                    f.write(f"Index:{idx}\n{text_conv}\n---\n")

        return cleaned_outputs


class FoundryLanguageModel:
    def __init__(
        self,
        seed: int,
        system_prompt: str,
        answer_regex: str,
        retry_prompt: str,
        endpoint: str,
        api_key: str,
        deployment_name: str = 'Llama-3.3-70B-Instruct',
        logdir: Optional[str] = None,
        annotator_string: Optional[str] = 'all_convs',
    ) -> None:
        self.answer_regex = answer_regex
        self.retry_prompt = retry_prompt
        self.client = OpenAI(
            base_url=f"{endpoint}",
            api_key=api_key,
        )
        self.deployment_name = deployment_name
        self.all_convs = ''
        self.system_prompt = system_prompt
        self.logdir = logdir
        if self.logdir is not None:
            os.makedirs(self.logdir, exist_ok=True)
        self.annotator_string = annotator_string
        self.retry_sleep_seconds = int(os.getenv("FOUNDRY_RETRY_SLEEP_SECONDS", "180"))
        self.max_retries = int(os.getenv("FOUNDRY_MAX_RETRIES", "0"))

    def _format_conversation(self, messages: List[dict]) -> str:
        lines = []
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _log_content_filter(self, index: int, message: str, error_details: Optional[str] = None) -> None:
        if self.logdir is None:
            return
        log_path = os.path.join(self.logdir, f"{self.annotator_string}_content_filter.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Index: {index}\n")
            if error_details:
                f.write("Error:\n")
                f.write(error_details)
                f.write("\n")
            f.write("Prompt:\n")
            f.write(message)
            f.write("\n---\n")

    def _format_content_filter_error(self, exc: Exception) -> str:
        body = getattr(exc, "body", None)
        if body is not None:
            return str(body)
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                return str(response.json())
            except Exception:
                return str(response)
        return str(exc)

    def _is_content_filter_error(self, exc: Exception) -> bool:
        if not isinstance(exc, BadRequestError):
            return False
        return "content_filter" in str(exc)

    def _is_retryable_error(self, exc: Exception) -> bool:
        return isinstance(
            exc,
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError),
        )

    def _request_with_retries(self, messages: List[dict], index: int):
        attempt = 0
        while True:
            try:
                return self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.1,
                    top_p=0.95,
                    max_tokens=4096,
                )
            except Exception as exc:
                if self._is_content_filter_error(exc):
                    raise
                if not self._is_retryable_error(exc):
                    raise
                attempt += 1
                if self.max_retries > 0 and attempt > self.max_retries:
                    raise
                print(
                    f"Retryable error at index {index}; sleeping {self.retry_sleep_seconds}s "
                    f"before retry {attempt}."
                )
                time.sleep(self.retry_sleep_seconds)

    def generate(self, messages: List[str], logging_indices: Sequence[int] = None, iteration: int = 0) -> List[int]:
        if logging_indices is not None:
            assert len(messages) == len(logging_indices)
        convs = []

        # Parse all the outputs
        cleaned_outputs = np.full(len(messages), AnnotationIdx.UNKOWN)
        for i, message in enumerate(messages):
            chat = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]
            try:
                response = self._request_with_retries(chat, i)
            except Exception as exc:
                if self._is_content_filter_error(exc):
                    print(f"Content filter triggered at index {i}; marking as content_filter.")
                    cleaned_outputs[i] = AnnotationIdx.CONTENT_FILTER
                    self._log_content_filter(i, message, self._format_content_filter_error(exc))
                    convs.append(chat)
                    continue
                raise
            text_answer = response.choices[0].message.content or ""
            chat.append({"role": "assistant", "content": text_answer})
            result = re.search(self.answer_regex, text_answer)
            if result:
                try:
                    best_sequence = int(result.group(1))
                    if best_sequence == 1:
                        best_sequence = AnnotationIdx.FIRST
                    elif best_sequence == 2:
                        best_sequence = AnnotationIdx.SECOND
                except ValueError:
                    best_sequence = AnnotationIdx.TIE
                cleaned_outputs[i] = best_sequence
            else:
                chat.append({"role": "user", "content": self.retry_prompt})
                try:
                    retry_response = self._request_with_retries(chat, i)
                except Exception as exc:
                    if self._is_content_filter_error(exc):
                        print(f"Content filter triggered at index {i} (retry); marking as content_filter.")
                        cleaned_outputs[i] = AnnotationIdx.CONTENT_FILTER
                        self._log_content_filter(i, message, self._format_content_filter_error(exc))
                        convs.append(chat)
                        continue
                    raise
                retry_text = retry_response.choices[0].message.content or ""
                chat.append({"role": "assistant", "content": retry_text})
                retry_result = re.search(self.answer_regex, retry_text)
                if retry_result:
                    try:
                        best_sequence = int(retry_result.group(1))
                        if best_sequence == 1:
                            best_sequence = AnnotationIdx.FIRST
                        elif best_sequence == 2:
                            best_sequence = AnnotationIdx.SECOND
                    except ValueError:
                        best_sequence = AnnotationIdx.TIE
                    cleaned_outputs[i] = best_sequence
            convs.append(chat)

        # Log the conversations
        if self.logdir is not None and logging_indices is not None:
            log_path = os.path.join(self.logdir, f"{self.annotator_string}.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                for conv, idx in zip(convs, logging_indices):
                    text_conv = self._format_conversation(conv)
                    f.write(f"Index:{idx}\n{text_conv}\n---\n")

        return cleaned_outputs

#! 追記
# class OllamaLanguageModel:
#     def __init__(
#         self,
#         seed: int,
#         system_prompt: str,
#         answer_regex: str,
#         retry_prompt: str,
#         model_name: str = 'ollama/llama3.1:8b-instruct-q3_K_S',
#         num_gpus: int = 8,
#         logdir: Optional[str] = None,
#         annotator_string: Optional[str] = 'all_convs',
#     ) -> None:

#         self.model_name = model_name
#         self.answer_regex = answer_regex
#         self.retry_prompt = retry_prompt
#         self.llm = ollama.chat(model=model_name, num_gpus=num_gpus, seed=seed)
#         self.all_convs = ''
#         self.system_prompt = system_prompt
#         self.logdir = logdir
#         if self.logdir is not None:
#             os.makedirs(self.logdir, exist_ok=True)
#         self.annotator_string = annotator_string

#     def generate(self, messages: List[str], logging_indices: Sequence[int] = None, iteration: int = 0) -> List[int]:
#         assert len(messages) == len(logging_indices)
#         prompts = []
#         convs = []

#         for message in messages:
#             conv = get_conversation_template(self.model_name)
#             conv.append_message(conv.roles[0], message)
#             conv.append_message(conv.roles[1], None)
#             conv.system = self.system_prompt
#             prompt = conv.get_prompt()
#             prompts.append(prompt)
#             convs.append(conv)

#         outputs = self.llm.generate(prompts)

#         # Parse all the outputs
#         cleaned_outputs = np.full(len(messages), AnnotationIdx.UNKOWN)
#         indexes_to_retry = []
#         prompts_to_retry = []
#         for i, output in enumerate(outputs):
#             text_answer = output.text
#             result = re.search(self.answer_regex, text_answer)
#             conv = convs[i]
#             conv.append_message('', text_answer)
#             if result:
#                 try:
#                     best_sequence = int(result.group(1))
#                     if best_sequence == 1:
#                         best_sequence = AnnotationIdx.FIRST
#                     elif best_sequence == 2:
#                         best_sequence = AnnotationIdx.SECOND
#                 except ValueError:
#                     best_sequence = AnnotationIdx.TIE
#                 cleaned_outputs[i] = best_sequence
#             else:
#                 # Ask the model again
#                 conv.append_message(conv.roles[0], self.retry_prompt)
#                 conv.append_message(conv.roles[1], None)
#                 prompt = conv.get_prompt()
#                 prompts_to_retry.append(prompt)
#                 indexes_to_retry.append(i)

#         # Retry the prompts that were not good
#         print("Retrying prompts")
#         second_batch = self.llm.generate(prompts_to_retry)
#         for i, output in zip(indexes_to_retry, second_batch):
#             text_answer = output.text
#             convs[i].append_message('', text_answer)
#             result = re.search(self.answer_regex, text_answer)
#             if result:
#                 try:
#                     best_sequence = int(result.group(1))
#                     if best_sequence == 1:
#                         best_sequence = AnnotationIdx.FIRST
#                     elif best_sequence == 2:
#                         best_sequence = AnnotationIdx.SECOND
#                 except ValueError:
#                     best_sequence = AnnotationIdx.TIE
#                 cleaned_outputs[i] = best_sequence

#         # Log the conversations
#         if self.logdir is not None and logging_indices is not None:
#             for conv, idx in zip(convs, logging_indices):
#                 text_conv = conv.get_prompt()
#                 self.all_convs += f" Index:{idx}\n {text_conv}\n"
#             with open(os.path.join(self.logdir, f"{self.annotator_string}.txt"), 'w') as f:
#                 f.write(self.all_convs)

#         return cleaned_outputs
