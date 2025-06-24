from openai import OpenAI
import threading
import sys
from io import StringIO
import time
import chat_session


class MultiChatSession:
    def __init__(self, ip, port, context_separator="###"):
        self.ip = ip
        self.port = port
        self.context_separator = context_separator

        self.chats = []

    # def get_openai_client(self, ip, port):
    #     openai_api_key = "EMPTY"
    #     openai_api_base = f"http://{ip}:{port}/v2"

    #     return OpenAI(
    #         # defaults to os.environ.get("OPENAI_API_KEY")
    #         api_key=openai_api_key,
    #         base_url=openai_api_base,
    #     )

    def init_chat(self):
        # client = self.get_openai_client(self.ip, self.port)
        # models = client.models.list()
        # model = models.data[0].id

        return chat_session.ChatSession(self.ip, self.port, self.context_separator)
        # self.chats.append(chat)
        # return chat

    def close(self):
        # self.client.close()
        self.chats.clear()
