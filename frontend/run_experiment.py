import sys
import random
import chat_session

from transformers import AutoTokenizer
from request_generator import RequestGenerator
from utils import (
    read_chunks,
    read_prompts,
    plot_latency_vs_seq_length,
    plot_multiple_latency_vs_seq_length,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000

SYSTEM_PROMPT = (
    "System prompt: You are a helpful assistant. I will now give you a document"
    "and please answer my question afterwards based on the content in document"
)


def parse_and_verify_args():
    if len(sys.argv) != 5:
        print(
            "Usage: python run_experiment.py <task> <num_contexts> <num_requests> <randomize>"
        )
        sys.exit(1)

    task = sys.argv[1]
    num_contexts = int(sys.argv[2])
    num_requests = int(sys.argv[3])
    randomize = sys.argv[4].lower() == "true"
    return task, num_contexts, num_requests, randomize


def get_context_keys(chunks, prompts, num_contexts, randomize):
    all_keys = list(chunks.keys())
    if randomize:
        context_keys = random.sample(all_keys, num_contexts)
    else:
        context_keys = all_keys[:num_contexts]

    for key in context_keys:
        if key not in prompts:
            print(f"Context key '{key}' not found in prompts.")
            sys.exit(1)

    return context_keys


def get_session_context_prompts_dict(
    task, tokenizer, chunks, prompts, context_keys, num_requests, randomize
):
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
    return session_context_prompts_dict


def main():
    task, num_contexts, num_requests, randomize = parse_and_verify_args()

    # Read data
    chunks = read_chunks("data/")
    prompts = read_prompts("prompts/")

    # Get context keys
    context_keys = get_context_keys(chunks, prompts, num_contexts, randomize)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Generate prompt-context tuples
    session_context_prompts_dict = get_session_context_prompts_dict(
        task, tokenizer, chunks, prompts, context_keys, num_requests, randomize
    )

    # Run requests
    generator = RequestGenerator(SYSTEM_PROMPT, session_context_prompts_dict)
    for _, _ in generator.start():
        pass

    # Plot latency vs sequence length
    plot_latency_vs_seq_length(task)
    plot_multiple_latency_vs_seq_length(task)


if __name__ == "__main__":
    main()
