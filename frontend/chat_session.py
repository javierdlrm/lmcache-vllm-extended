from openai import OpenAI
from io import StringIO
import time
import csv
import os


class ChatSession:
    def __init__(
        self, ip, port, context_separator="###", tokenizer=None, task="thetask"
    ):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://{ip}:{port}/v2"

        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        self.model = models.data[0].id

        self.messages = []

        self.final_context = ""
        self.context_separator = context_separator
        self.tokenizer = tokenizer
        self.task = task

    def set_context(self, context_strings):
        contexts = []
        for context in context_strings:
            contexts.append(context)

        self.final_context = self.context_separator.join(contexts)
        self.on_user_message(self.final_context, display=False)
        self.on_server_message("Got it!", display=False)

    def get_context(self):
        return self.final_context

    def on_user_message(self, message, display=True):
        if display:
            print("👤 User message:", message, end="\n\n")
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message, display=True):
        if display:
            print("🤖 Server message:", message, end="\n\n")
        self.messages.append({"role": "assistant", "content": message})

    def chat(self, question):
        self.on_user_message(question)

        num_char, seq_length = self.get_num_char_and_seq_length(self.messages)

        start = time.perf_counter()
        end = None

        print("----------------------------------------------------------------------")
        print(f"# Messages [{len(self.messages)} messages]:", self.messages)
        print(
            "----------------------------------------------------------------------",
            end="\n\n",
        )

        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=0.5,
            stream=True,
            stop="\n",
        )

        output_buffer = StringIO()
        server_message = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if end is None:
                    end = time.perf_counter()
                yield chunk_message
                server_message.append(chunk_message)

        self.on_server_message("".join(server_message))

        latency = end - start

        # save metrics to csv file
        self.record_response_metrics(seq_length, latency)

        print(
            f"\n\n(📝 Response delay: {latency:.2f} seconds // {num_char} chars, {seq_length} tokens (seq len))\n"
        )

    def get_num_char_and_seq_length(self, messages):
        chat_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
        token_ids = self.tokenizer.encode(chat_str)
        return len(chat_str), len(token_ids)

    def record_response_metrics(self, seq_length, latency):
        csv_file = f"reports/{self.task}.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["seq_length", "latency"])
            writer.writerow([seq_length, latency])
