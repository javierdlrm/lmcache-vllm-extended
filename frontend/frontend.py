import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

# Change the following variables as needed
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000


@st.cache_resource
def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


@st.cache_data
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames
    """
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

@st.cache_data
def read_prompts(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames
    """
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


chunks = read_chunks("data/")
selected_chunks = st.multiselect(
    "Select the chunks into the context",
    list(chunks.keys()),
    default=[],
    placeholder="Select in the drop-down menu",
)
contexts = [chunks[key] for key in selected_chunks]

prompts = read_prompts("prompts/question1")

container = st.container(border=True)

with st.sidebar:
    system_prompt = st.text_area(
        "System prompt:",
        "You are a helpful assistant. I will now give you a document and "
        "please answer my question afterwards based on the content in document",
    )

    session = chat_session.ChatSession(IP1, PORT1)
    session.set_context([system_prompt] + contexts)

    print("New session created with context")

    num_tokens = tokenizer.encode(session.get_context())
    container.header(
        f"The context given to LLM: ({len(num_tokens)} tokens)", divider="grey"
    )
    container.text(session.get_context())

    messages = st.container(height=300)
    messages.markdown("*vLLM instance 1*")
    if prompt := st.chat_input("Type the question here", key=1):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))

    st.text("🚀 Request generator:")
    num_requests = st.number_input(
        "Select number of requests:",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        key="num_requests",
    )

    if st.button("Start", key="send_multichat"):
        print(f"Selected_chunks [{len(selected_chunks)}]: {selected_chunks}")

        if num_requests > 0 and len(selected_chunks) > 0:
            context_and_prompts = []
            for key in selected_chunks:
                session_context = [chunks[key]]
                for prompt in prompts[key]:
                    context_and_prompts.append((session_context, prompt))

            # randomized_ctx_and_prompts = [
            #     random.choice(context_and_prompts) for _ in range(num_requests)
            # ]
            randomized_ctx_and_prompts = context_and_prompts[:num_requests]
            # print(f"Randomized ctx and prompts [{len(randomized_ctx_and_prompts)}]")
            
            for i, ctx_prompt in enumerate(randomized_ctx_and_prompts):
                session_context, prompt = ctx_prompt

                # For each iteration, append session_context i times (first iteration: once, then keep increasing)
                repeated_context = [system_prompt] + session_context * (i + 1)

                session = chat_session.ChatSession(IP1, PORT1)
                session.set_context(repeated_context)

                # chat = multichat.init_chat()
                # chat.set_context([system_prompt] + session_context)
                messages.chat_message("user").write(f"[Seq-length: {len(session.get_context())}] " + prompt)
                messages.chat_message("assistant").write_stream(session.chat(prompt))