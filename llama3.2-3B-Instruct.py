# Llama-3.2-3B-Instruct.py

import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
print()

# 生成されたメッセージ全体を取得
generated_messages = outputs[0]["generated_text"]

# 最後のメッセージ（アシスタントの返答）を取得
assistant_message = generated_messages[-1]

# "content" のみを表示
print(assistant_message["content"])
