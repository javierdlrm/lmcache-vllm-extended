import sys
import os
from transformers import AutoTokenizer
from request_generator import RequestGenerator
import chat_session
import random

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


def main():
    if len(sys.argv) != 4:
        print("Usage: python run_experiment.py <task> <num_contexts> <num_requests>")
        sys.exit(1)

    task = sys.argv[1]
    num_contexts = int(sys.argv[2])
    num_requests = int(sys.argv[3])

    # Read data
    chunks = read_chunks("data/")
    prompts = read_prompts("prompts/")

    num_contexts = int(num_contexts)
    if num_contexts > len(chunks):
        print(
            f"Requested num_contexts ({num_contexts}) exceeds available chunks ({len(chunks)})."
        )
        sys.exit(1)

    context_keys = random.sample(list(chunks.keys()), num_contexts)

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
        selected_prompts = prompts[context_key][:num_requests]
        session_context_prompts_dict[context_key] = {
            "session": chat_session.ChatSession(
                IP1, PORT1, tokenizer=tokenizer, task=task
            ),
            "context": chunks[context_key],
            "prompts": selected_prompts,
        }

    # # Prepare prompt-context tuples
    # selected_prompts = prompts[context_key][:num_requests]
    # session_context = [chunks[context_key]]
    # # prompt_context_tuples = [(prompt, session_context) for prompt in selected_prompts]

    # Run requests
    generator = RequestGenerator(session_context_prompts_dict)
    current_prompt = None
    for prompt, response_chunk in generator.start():
        if current_prompt != prompt:
            current_prompt = prompt
            print(f"Prompt: {prompt}\n")
        print(f"\rResponse chunk: {response_chunk}", end="", flush=True)


if __name__ == "__main__":
    main()
