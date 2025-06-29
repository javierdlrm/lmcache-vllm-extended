import os
import matplotlib.pyplot as plt
import csv
import glob


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
            latencies.append(float(row["latency"]) * 1000)  # milliseconds

    if not seq_lengths or not latencies:
        print("No data to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(seq_lengths, latencies, color="blue", alpha=0.7, label="Samples")
    # Sort by sequence length for a meaningful line
    sorted_pairs = sorted(zip(seq_lengths, latencies))
    sorted_seq_lengths, sorted_latencies = zip(*sorted_pairs)
    plt.plot(sorted_seq_lengths, sorted_latencies, color="orange", label="Trend")
    plt.title(f"Latency vs Sequence Length for task: {task}")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Latency (milliseconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = f"reports/{task}_latency_vs_seq_length.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_multiple_latency_vs_seq_length(task):
    plt.figure(figsize=(10, 6))

    task_prefix = task[:15]
    pattern = f"reports/{task_prefix}_*.csv"
    csv_paths = glob.glob(pattern)
    if not csv_paths:
        print(f"No CSV files found for pattern: {pattern}")
        return

    labels = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
    title = f"Latency vs Sequence Length for task: {task}"

    for idx, csv_file in enumerate(csv_paths):
        seq_lengths = []
        latencies = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_lengths.append(int(row["seq_length"]))
                latencies.append(float(row["latency"]) * 1000)  # milliseconds

        # Sort for a meaningful line
        sorted_pairs = sorted(zip(seq_lengths, latencies))
        sorted_seq_lengths, sorted_latencies = zip(*sorted_pairs)
        label = labels[idx] if labels and idx < len(labels) else f"Run {idx+1}"
        plt.plot(sorted_seq_lengths, sorted_latencies, marker="o", label=label)

    plt.title(title)
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Latency (milliseconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = "reports/multi_latency_vs_seq_length.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Multi-line plot saved to {output_path}")
