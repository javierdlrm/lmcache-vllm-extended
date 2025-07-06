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


def plot_latency_vs_req_id(task):
    csv_file = f"reports/{task}.csv"
    if not os.path.isfile(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return

    req_ids = []
    latencies = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        req_id = 1
        for row in reader:
            req_ids.append(req_id)
            req_id += 1
            latencies.append(float(row["latency"]) * 1000)  # milliseconds

    if not req_ids or not latencies:
        print("No data to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(req_ids, latencies, color="blue", alpha=0.7, label="Samples")
    # Sort by sequence length for a meaningful line
    # sorted_pairs = sorted(zip(seq_lengths, latencies))
    sorted_pairs = sorted(zip(req_ids, latencies))
    sorted_seq_lengths, sorted_latencies = zip(*sorted_pairs)
    plt.plot(sorted_seq_lengths, sorted_latencies, color="orange", label="Trend")
    plt.title(f"Latency vs Request ID for task: {task}")
    plt.xlabel("Request ID")
    plt.ylabel("Latency (milliseconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = f"reports/{task}_latency_vs_req_id.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_rag_latency_vs_req_id(task):
    csv_file = f"reports/{task}.csv"
    if not os.path.isfile(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return

    req_ids = []
    rag_encode_latencies = []
    rag_search_latencies = []
    rag_latencies = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        req_id = 1
        for row in reader:
            req_ids.append(req_id)
            req_id += 1
            # Safely get each latency, default to 0 if missing or empty
            rag_encode_lat = float(row.get("rag_encode_latency", 0) or 0) * 1000
            rag_search_lat = float(row.get("rag_search_latency", 0) or 0) * 1000
            rag_lat = float(row.get("rag_latency", 0) or 0) * 1000
            rag_encode_latencies.append(rag_encode_lat)
            rag_search_latencies.append(rag_search_lat)
            rag_latencies.append(rag_lat)

    if not req_ids or not rag_latencies:
        print("No data to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(
        req_ids,
        rag_encode_latencies,
        color="blue",
        alpha=0.7,
        label="RAG Encode Latency",
    )
    plt.plot(
        req_ids,
        rag_search_latencies,
        color="orange",
        alpha=0.7,
        label="RAG Search Latency",
    )
    plt.plot(
        req_ids, rag_latencies, color="green", alpha=0.7, label="RAG Total Latency"
    )
    plt.title(f"RAG Latencies vs Request ID for task: {task}")
    plt.xlabel("Request ID")
    plt.ylabel("Latency (milliseconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = f"reports/{task}_rag_latency_vs_req_id.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_multiple_latency_vs_seq_length(task):
    plt.figure(figsize=(10, 6))

    task_prefix = task[:15]
    pattern = f"reports/{task_prefix}_*.csv"
    csv_paths = [p for p in glob.glob(pattern) if not p.endswith("_rag.csv")]
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
    output_path = f"reports/{task}_multi_latency_vs_seq_length.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Multi-line plot saved to {output_path}")


def plot_multiple_rag_latency_vs_req_id(task):
    plt.figure(figsize=(10, 6))

    task_prefix = task[: len("rag_benchmark")]
    pattern = f"reports/{task_prefix}_*.csv"
    csv_paths = glob.glob(pattern)
    if not csv_paths:
        print(f"No CSV files found for pattern: {pattern}")
        return

    labels = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
    title = f"RAG Latency vs Request ID for task: {task}"

    for idx, csv_file in enumerate(csv_paths):
        req_ids = []
        rag_latencies = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            req_id = 1
            for row in reader:
                if "rag_latency" in row and row["rag_latency"]:
                    req_ids.append(req_id)
                    rag_latencies.append(float(row["rag_latency"]) * 1000)  # ms
                    req_id += 1
        if req_ids and rag_latencies:
            label = labels[idx] if labels and idx < len(labels) else f"Run {idx+1}"
            plt.plot(req_ids, rag_latencies, marker="o", label=label)

    plt.title(title)
    plt.xlabel("Request ID")
    plt.ylabel("RAG Latency (milliseconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = f"reports/{task}_multi_rag_latency_vs_req_id.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Multi-line RAG latency plot saved to {output_path}")


def get_num_char_and_seq_length(tokenizer, messages):
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer.encode(chat_str)
    return len(chat_str), len(token_ids)


def record_response_metrics(task, values, header=["seq_length", "latency"], suffix=""):
    csv_file = f"reports/{task}{suffix}.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(values)
