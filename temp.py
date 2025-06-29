class RequestGenerator:
    def __init__(self, chat_session, prompt_context_tuples):
        """
        :param chat_session: An instance of ChatSession
        :param prompt_context_tuples: List of tuples (prompt, session_context)
        """
        self.chat_session = chat_session
        self.prompt_context_tuples = prompt_context_tuples

    def start(self):
        """
        Iterates over the list of (prompt, session_context) tuples,
        sets the context, and streams the chat responses.
        Yields (prompt, response_chunk) for each chunk in the stream.
        """
        for prompt, session_context in self.prompt_context_tuples:
            self.chat_session.set_context(session_context)
            response_stream = self.chat_session.chat(prompt)
            for chunk in response_stream:
                yield (prompt, chunk)