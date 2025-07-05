import random
import time
import requests
from utils import record_response_metrics, get_num_char_and_seq_length


class RequestGenerator:

    def __init__(
        self, system_prompt, session_context_prompts_dict, ip, port, task, tokenizer
    ):
        self.system_prompt = system_prompt
        self.session_context_prompts_dict = session_context_prompts_dict
        self.ip = ip
        self.port = port
        self.task = task
        self.tokenizer = tokenizer

    def start(self, randomize=True):
        # All session and prompts combinations
        session_prompt_tuples = []

        for entry in self.session_context_prompts_dict.values():
            sessions = entry["sessions"]
            context = entry["context"]
            prompts = entry["prompts"]
            context_key = entry["context_key"]

            for session, prompt in zip(sessions, prompts):
                session.set_context_key(context_key)
                session.set_context([context])
                session_prompt_tuples.append((session, prompt))

        # Randomize the order of session and prompt combinations
        if randomize:
            random.shuffle(session_prompt_tuples)

        # For each session and prompt, yield the response stream
        for session, prompt in session_prompt_tuples:
            response_stream = session.chat(prompt)
            for response_chunk in response_stream:
                yield prompt, response_chunk

    def start_batch(self, randomize=True, sort_before_forwarding=True, use_rag=False):
        # All session and prompts combinations
        session_prompt_tuples = []

        for entry in self.session_context_prompts_dict.values():
            sessions = entry["sessions"]
            context = entry["context"]
            prompts = entry["prompts"]
            context_key = entry["context_key"]

            for session, prompt in zip(sessions, prompts):
                session.set_context_key(context_key)
                if not use_rag:  # If not using RAG, set the session context
                    session.set_context([context])
                session_prompt_tuples.append((session, prompt))

        # Randomize the order of session and prompt combinations
        if randomize:
            random.shuffle(session_prompt_tuples)

        # For each session and prompt, build a request
        request_batch = []
        for session, prompt in session_prompt_tuples:
            request = session.build_chat_completion_request(prompt)
            request_batch.append(request)

        return self._send_batch(
            request_batch,
            sort_before_forwarding=sort_before_forwarding,
            use_rag=use_rag,
        )

    def _send_batch(self, request_batch, sort_before_forwarding=True, use_rag=False):
        print("----------------------------------------------------------------------")
        print(f"# Batch size: [{len(request_batch)}]:")
        print(
            "----------------------------------------------------------------------",
            end="\n\n",
        )

        start = time.perf_counter()
        end = None

        batch_payload = {
            "requests": request_batch,
            "sort_before_forwarding": sort_before_forwarding,
            "use_rag": use_rag,
        }

        print("\n\n--------------------------------------------------------------")
        print("# Request:")
        print(batch_payload)
        print("--------------------------------------------------------------\n\n")

        url = f"http://{self.ip}:{self.port}/v2" + "/batch/chat/completions"
        response = requests.post(url, json=batch_payload)

        end = time.perf_counter()
        latency = end - start

        print(f"\n\n(📝 Response delay: {latency:.2f} seconds\n")

        response.raise_for_status()
        response_json = response.json()

        print("\n\n--------------------------------------------------------------")
        print("# Response:")
        print(response_json)
        print("--------------------------------------------------------------\n\n")

        # save metrics to csv file
        for metrics in response_json:
            header, values = [], []
            if use_rag:
                print("!!! ////// Recompute seq length with: ", metrics["messages"])
                _, seq_length = get_num_char_and_seq_length(
                    self.tokenizer, metrics["messages"]
                )
            else:
                seq_length = metrics["seq_length"]

            header.append("seq_length")
            header.append("latency")
            values.append(seq_length)
            values.append(metrics["latency"])

            if use_rag:
                header.append("rag_accuracy")
                header.append("rag_latency")
                values.append(metrics["rag_accuracy"])
                values.append(metrics["rag_latency"])

            record_response_metrics(self.task, values, header=header)

        return response_json

    def index_contexts_for_rag(self):
        url = f"http://{self.ip}:{self.port}/v2/rag/index"

        print("\n/// Indexing contexts for RAG...")

        start = time.perf_counter()
        end = None

        for entry in self.session_context_prompts_dict.values():
            context_key = entry["context_key"]
            context = entry["context"]
            payload = {
                "context_key": context_key,
                "context": context,
            }
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                print(f"- Indexed context '{context_key}': {response.json()}")
            except Exception as e:
                print(f"Failed to index context '{context_key}': {e}")

        end = time.perf_counter()
        latency = end - start

        print(f"\n\n(📝 Indexing delay: {latency:.2f} seconds\n")
