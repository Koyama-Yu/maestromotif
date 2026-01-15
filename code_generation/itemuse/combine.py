import os
import shutil
import sys


def extract_code(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    
    extracted_code = []
    for line in lines:
        if "# starting conditions" in line.strip().lower():
            break
        extracted_code.append(line)

    return ''.join(extracted_code)

def save_combined_code(combined_code, output_path):
    """
    Saves the combined code to the specified output path.
    """
    with open(output_path, 'w') as file:
        file.write(combined_code)

def main():
    seed = 42 if len(sys.argv) != 2 else sys.argv[1]

    path = f'_seed{seed}/execution.py'
    os.makedirs(f'seed{seed}', exist_ok=True)
    output_path = f'seed{seed}/execution.py'
    
    combined_code = extract_code(path)

    with open('termination_itemuse.txt', 'r') as file:
        lines = file.readlines()
        term = "".join(lines)
    combined_code += term

    with open('precondition_itemuse.txt', 'r') as file:
        lines = file.readlines()
        precon = "".join(lines)
    combined_code += precon

    with open('initial_values.txt', 'r') as file:
        lines = file.readlines()
        initval = "".join(lines)
    combined_code += initval

    with open(output_path, 'w') as file:
        file.write(combined_code)
    with open(f'seed{seed}/__init__.py', 'w') as file:file.write('')

    print(f"Combined code has been saved to {output_path}")



if __name__ == "__main__":
    main()
