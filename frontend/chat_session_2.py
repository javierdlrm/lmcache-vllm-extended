from openai import OpenAI
import threading
import sys
from io import StringIO
import time


class ChatSession:
    def __init__(self, client, model, context_separator="###"):
        self.client = client
        self.model = model

        self.messages = []

        self.final_context = ""
        self.context_separator = context_separator

    def set_context(self, context_strings):

        print("[set_context] Context strings: ", context_strings, end="\n\n")

        contexts = []
        for context in context_strings:
            contexts.append(context)

        self.final_context = self.context_separator.join(contexts)
        self.on_user_message(self.final_context, display=False)
        self.on_server_message("Got it!", display=False)

    def get_context(self):
        print("[get_context] Context: ", self.final_context, end="\n\n")
        return self.final_context

    def on_user_message(self, message, display=True):
        if display:
            print("[User message] ", message)
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message, display=True):
        if display:
            print("[Server message] ", message)
        self.messages.append({"role": "assistant", "content": message})

    def chat(self, question):
        self.on_user_message(question)

        start = time.perf_counter()
        end = None

        print("[Chat] Messages: ", self.messages, end="\n\n")

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
        yield f"\n\n(Response delay: {end - start:.2f} seconds)"
