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

    def start(self, randomize):
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

    def start_batch(self, randomize, sort_before_forwarding, use_rag):
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
        for idx, (session, prompt) in enumerate(session_prompt_tuples):
            request = session.build_chat_completion_request(prompt)
            request["request_idx"] = idx  # add idx for future reference
            request_batch.append(request)

        return self._send_batch(
            request_batch,
            sort_before_forwarding=sort_before_forwarding,
            use_rag=use_rag,
        )

    def _send_batch(self, request_batch, sort_before_forwarding, use_rag):
        url = f"http://{self.ip}:{self.port}/v2" + "/batch/chat/completions"

        batch_payload = {
            "requests": request_batch,
            "sort_before_forwarding": sort_before_forwarding,
            "use_rag": use_rag,
        }

        start = time.perf_counter()
        end = None

        response = requests.post(url, json=batch_payload)

        end = time.perf_counter()
        latency = end - start

        print(f"\n(📝 Response delay: {latency:.2f} seconds\n")

        response.raise_for_status()
        response_json = response.json()

        print("--------------------------------------------------------------")
        print("# Response:")
        print(response_json)
        print("--------------------------------------------------------------\n")

        # save metrics to csv file
        for metrics in response_json:
            header, values = [], []

            request_idx = metrics["request_idx"]
            request = request_batch[request_idx]  # find original request

            _, seq_length = get_num_char_and_seq_length(
                self.tokenizer, request["request"]["messages"]
            )
            # seq_length = metrics["seq_length"]

            header.append("seq_length")
            values.append(seq_length)
            header.append("latency")
            values.append(metrics["latency"])

            record_response_metrics(self.task, values, header=header)

            if use_rag:
                header_rag, values_rag = [], []
                header_rag.append("rag_accuracy")
                values_rag.append(metrics["rag_accuracy"])
                header_rag.append("rag_latency")
                values_rag.append(metrics["rag_latency"])
                header_rag.append("rag_match")
                values_rag.append(
                    1 if metrics["rag_context_key"] == request["context_key"] else 0
                )
                header_rag.append("context_key")
                values_rag.append(request["context_key"])
                header_rag.append("rag_context_key")
                values_rag.append(metrics["rag_context_key"])
                header_rag.append("prompt")
                values_rag.append(request["request"]["messages"][0]["content"])

                record_response_metrics(
                    self.task, values_rag, header=header_rag, suffix="_rag"
                )

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
                print(f"... Indexed context '{context_key}': {response.json()}")
            except Exception as e:
                print(f"Failed to index context '{context_key}': {e}")

        end = time.perf_counter()
        latency = end - start

        print(f"\n(📝 Indexing delay: {latency:.2f} seconds\n")

    def rag_benchmark(self):
        url = f"http://{self.ip}:{self.port}/v2/rag/search"

        print("\n/// Searching contexts for RAG...")

        accuracy = 0
        total = 0

        for entry in self.session_context_prompts_dict.values():
            prompts = entry["prompts"]
            context_key = entry["context_key"]

            for prompt in prompts:
                payload = {"prompt": prompt}

                start = time.perf_counter()
                end = None

                try:
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    print(f"... Found context '{context_key}': {response.json()}")
                except Exception as e:
                    print(f"Failed to find context '{context_key}': {e}")

                end = time.perf_counter()
                latency = end - start

                response_json = response.json()
                rag_match = 1 if response_json["rag_context_key"] == context_key else 0
                accuracy += rag_match
                total += 1

                header_rag, values_rag = [], []
                # header_rag.append("rag_accuracy")
                # values_rag.append(metrics["rag_accuracy"])
                header_rag.append("rag_latency")
                values_rag.append(latency)
                header_rag.append("rag_match")
                values_rag.append(rag_match)
                header_rag.append("context_key")
                values_rag.append(context_key)
                header_rag.append("rag_context_key")
                values_rag.append(response_json["context_key"])
                header_rag.append("prompt")
                values_rag.append(prompt)

                record_response_metrics(
                    self.task, values_rag, header=header_rag, suffix="_rag"
                )

                if total > 0:
                    print(f"# Accuracy: {(accuracy/total) * 100}%")
                else:
                    print("No questions found")
