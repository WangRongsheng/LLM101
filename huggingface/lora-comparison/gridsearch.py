import itertools
import subprocess
import threading
import time

import torch

num_devices = torch.cuda.device_count()
if num_devices < 1:
    print("Please select a machine with at least 1 GPU.")
    quit()

alpha_values = [1, 4, 8, 16, 32, 64]
rank_values = [1, 2, 4, 8, 16, 32]
devices = range(num_devices)
lora_query = ["True"]
lora_key = ["False", "True"]
lora_value = ["True"]
lora_projection = ["False", "True"]
lora_mlp = ["False", "True"]
lora_head = ["False", "True"]

# Dictionary to keep track of whether a device is currently in use
device_usage = {device: False for device in devices}

# Set to keep track of used hyperparameter combinations
used_combinations = set()

def run_script(alpha, rank, device, query, key, value, projection, mlp, head):
    global device_usage

    command = [
        'python', 'finetune-lora-script.py',
        '--lora_alpha', str(alpha),
        '--lora_r', str(rank),
        '--device', str(device),
        '--lora_query', query,
        '--lora_key', key,
        '--lora_value', value,
        '--lora_projection', projection,
        '--lora_mlp', mlp,
        '--lora_head', head,
        '--verbose', "False"
    ]

    print(f"Starting run with alpha = {alpha}, rank = {rank}, lora_query = {query}, lora_key = {key}, lora_value = {value}, lora_projection = {projection}, lora_mlp = {mlp}, lora_head = {head} on device {device}")
    subprocess.run(command)
    print(f"Completed run with alpha = {alpha}, rank = {rank}, lora_query = {query}, lora_key = {key}, lora_value = {value}, lora_projection = {projection}, lora_mlp = {mlp}, lora_head = {head} on device {device}")

    # Mark the device as no longer in use
    device_usage[device] = False


def get_available_device():
    while True:
        for device, in_use in device_usage.items():
            if not in_use:
                device_usage[device] = True
                return device
        time.sleep(10)  # Wait before checking again


threads = []

# Using itertools.product to create combinations
for params in itertools.product(alpha_values, rank_values, lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head):
    alpha, rank, query, key, value, projection, mlp, head = params

    # Check if the combination has already been used
    if (alpha, rank, query, key, value, projection, mlp, head) in used_combinations:
        continue  # Skip this combination as it's already used

    # Mark the combination as used
    used_combinations.add((alpha, rank, query, key, value, projection, mlp, head))

    device = get_available_device()
    thread = threading.Thread(target=run_script, args=(alpha, rank, device, query, key, value, projection, mlp, head))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All runs completed.")