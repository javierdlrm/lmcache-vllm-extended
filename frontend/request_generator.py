import random
import time
import requests
from utils import record_response_metrics


class RequestGenerator:

    def __init__(self, system_prompt, session_context_prompts_dict, ip, port, task):
        self.system_prompt = system_prompt
        self.session_context_prompts_dict = session_context_prompts_dict
        self.ip = ip
        self.port = port
        self.task = task

    def start(self, randomize=True):
        # All session and prompts combinations
        session_prompt_tuples = []

        for entry in self.session_context_prompts_dict.values():
            session = entry["session"]
            context = entry["context"]
            prompts = entry["prompts"]
            context_key = entry["context_key"]

            extended_context = [context]
            session.set_context(extended_context, context_key=context_key)

            for prompt in prompts:
                session_prompt_tuples.append((session, prompt))

        # Randomize the order of session and prompt combinations
        if randomize:
            random.shuffle(session_prompt_tuples)

        # For each session and prompt, yield the response stream
        for session, prompt in session_prompt_tuples:
            response_stream = session.chat(prompt)
            for response_chunk in response_stream:
                yield prompt, response_chunk

    def start_batch(self, randomize=True, sort_before_forwarding=True):
        # All session and prompts combinations
        session_prompt_tuples = []

        for entry in self.session_context_prompts_dict.values():
            session = entry["session"]
            context = entry["context"]
            prompts = entry["prompts"]
            context_key = entry["context_key"]

            extended_context = [context]
            session.set_context(extended_context, context_key=context_key)

            for prompt in prompts:
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
            request_batch, sort_before_forwarding=sort_before_forwarding
        )

    def _send_batch(self, request_batch, sort_before_forwarding=True):
        print("----------------------------------------------------------------------")
        print(f"# Batch size: [{len(request_batch)}]:")
        print(
            "----------------------------------------------------------------------",
            end="\n\n",
        )

        start = time.perf_counter()
        end = None

        # Prepare the batch request payload
        batch_payload = {
            "requests": request_batch,
            "sort_before_forwarding": sort_before_forwarding,
        }

        url = f"http://{self.ip}:{self.port}/v2" + "/batch/chat/completions"
        response = requests.post(url, json=batch_payload)

        end = time.perf_counter()
        latency = end - start

        print(f"\n\n(📝 Response delay: {latency:.2f} seconds\n")

        response.raise_for_status()
        response_json = response.json()

        print("----------------------------------------------------------------------")
        print("# Response:")
        print(response_json)
        print("----------------------------------------------------------------------")

        # save metrics to csv file
        for metrics in response_json:
            seq_length = metrics["seq_length"]
            latency = metrics["latency"]
            record_response_metrics(self.task, seq_length, latency)

        return response_json
