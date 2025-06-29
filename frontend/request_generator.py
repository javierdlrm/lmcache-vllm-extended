class RequestGenerator:
    def __init__(self, session_context_prompts_dict):
        """
        :param session_context_prompts_dict: Dictionary with sessions, session contexts and prompts.
        """
        self.session_context_prompts_dict = session_context_prompts_dict

    def start(self):
        """
        For each (session, context, prompts) dict, set the context and yield the streamed response.
        """
        for entry in self.session_context_prompts_dict.values():
            session = entry["session"]
            context = entry["context"]
            prompts = entry["prompts"]

            session.set_context([context])
            for prompt in prompts:
                response_stream = session.chat(prompt)
                for response_chunk in response_stream:
                    yield prompt, response_chunk
