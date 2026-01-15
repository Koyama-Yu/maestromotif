import os 
import shutil
import subprocess
import sys

from huggingface_hub import InferenceClient


client = InferenceClient(
    token="your_token",
)

temperature = 0.2
messages = [{"role": "system", "content": "You are a helpful assistant."},]
model = "meta-llama/Meta-Llama-3.1-405B-Instruct"

seed = 42 if len(sys.argv) != 2 else sys.argv[1]

with open(f"../exploration/seed{seed}/execution.py", 'r') as f:
    previous_code = f.read()

previous_code = previous_code.split('def perform_task')[0]
previous_code += 'def perform_task(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):'
print(previous_code)

# initial message
with open("initial.txt", "r") as f:
    initial = f.read()
initial = initial.format(previous_code)
messages.append({"role": "user", "content": initial})

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=10000,
)
output = completion.choices[0].message.content
messages.append({"role": "assistant", "content": output})

try:
    code = output.split('```python')[1].split('```')[0]
    assert 'for turn in range' in code
    with open("execution.py", 'w') as f: f.write(code)
    result = subprocess.run(['python', "execution.py"], capture_output=True, text=True)
except:
    print("\n\n\nthere is a problem...")
    input()

print(result.stdout)

# code checking
with open("code_check.txt", 'r') as f:
    code_check = f.read()
code_check = code_check.format(result.stdout)
messages.append({"role": "user", "content": code_check})
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=10_000,
)
output = completion.choices[0].message.content
print(output)
messages.append({"role": "assistant", "content": output})
input()

iterations = 1
while 'yes' not in output.lower() and iterations < 4:
    print("\nimprovement iteration ", iterations, "\n")

    # simulate and rewrite
    with open("simulate_rewrite.txt", 'r') as f:
        simulate_rewrite = f.read()
    messages.append({"role": "user", "content": simulate_rewrite})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=10_000,
    )
    output = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": output})

    try:
        code = output.split('```python')[1].split('```')[0]
        assert 'for turn in range' in code
        with open("execution.py", 'w') as f: f.write(code)
        result = subprocess.run(['python', "execution.py"], capture_output=True, text=True)
    except:
        print("\n\n\nthere is a problem...")
        input()

    print(result.stdout)

    # output checking
    with open("code_check.txt", 'r') as f:
        code_check = f.read()
    code_check = code_check.format(result.stdout)   
    messages.append({"role": "user", "content": code_check})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=10_000,
    )
    output = completion.choices[0].message.content
    print(output)
    messages.append({"role": "assistant", "content": output})
    input()

    iterations += 1

### Saving important info

seed = 42 if len(sys.argv) != 2 else sys.argv[1]

source_dir = os.getcwd()
destination_dir = os.path.join(source_dir, f'_seed{seed}')
os.makedirs(destination_dir, exist_ok=True)

item_path = os.path.join(source_dir, 'execution.py')

# Check if the item is a file (not a directory)
if os.path.isfile(item_path):
    shutil.copy(item_path, destination_dir)
