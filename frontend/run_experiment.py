import sys
import os
from transformers import AutoTokenizer
from request_generator import RequestGenerator
import chat_session
import random
import matplotlib.pyplot as plt
import csv

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000


def read_chunks(file_folder):
    filenames = os.listdir(file_folder)
    ret = {}
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        key = filename.removesuffix(".txt")
        with open(os.path.join(file_folder, filename), "r") as fin:
            value = fin.read()
        ret[key] = value
    return ret


def read_prompts(file_folder):
    filenames = os.listdir(file_folder)
    ret = {}
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        key = filename.removesuffix(".txt")
        with open(os.path.join(file_folder, filename), "r") as fin:
            value = fin.read().splitlines()
        ret[key] = value
    return ret


def plot_latency_vs_seq_length(task):
    csv_file = f"reports/{task}.csv"
    if not os.path.isfile(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return

    seq_lengths = []
    latencies = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_lengths.append(int(row["seq_length"]))
            latencies.append(float(row["latency"]))

    if not seq_lengths or not latencies:
        print("No data to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(seq_lengths, latencies, color="blue", alpha=0.7)
    plt.title(f"Latency vs Sequence Length for task: {task}")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Latency (seconds)")
    plt.grid(True)
    plt.tight_layout()
    output_path = f"reports/{task}_latency_vs_seq_length.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python run_experiment.py <task> <num_contexts> <num_requests> <randomize>"
        )
        sys.exit(1)

    task = sys.argv[1]
    num_contexts = int(sys.argv[2])
    num_requests = int(sys.argv[3])
    randomize = sys.argv[4].lower() == "true"

    # Read data
    chunks = read_chunks("data/")
    prompts = read_prompts("prompts/")

    num_contexts = int(num_contexts)
    if num_contexts > len(chunks):
        print(
            f"Requested num_contexts ({num_contexts}) exceeds available chunks ({len(chunks)})."
        )
        sys.exit(1)

    all_keys = list(chunks.keys())
    if randomize:
        context_keys = random.sample(all_keys, num_contexts)
    else:
        context_keys = all_keys[:num_contexts]

    for key in context_keys:
        if key not in prompts:
            print(f"Context key '{key}' not found in prompts.")
            sys.exit(1)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Generate prompt-context tuples
    session_context_prompts_dict = {}
    for context_key in context_keys:
        if context_key not in prompts:
            print(f"Context key '{context_key}' not found in prompts.")
            sys.exit(1)
        prompt_list = prompts[context_key]
        if randomize:
            selected_prompts = random.sample(
                prompt_list, min(num_requests, len(prompt_list))
            )
        else:
            selected_prompts = prompt_list[:num_requests]
        session_context_prompts_dict[context_key] = {
            "session": chat_session.ChatSession(
                IP1, PORT1, tokenizer=tokenizer, task=task
            ),
            "context": chunks[context_key],
            "prompts": selected_prompts,
        }

    # Run requests
    generator = RequestGenerator(session_context_prompts_dict)
    for _, _ in generator.start():
        pass

    # Plot latency vs sequence length
    plot_latency_vs_seq_length(task)


if __name__ == "__main__":
    main()
