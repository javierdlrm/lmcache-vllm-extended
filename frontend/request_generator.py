class RequestGenerator:
    def __init__(self, session_context_prompts_dict):
        """
        :param session_context_prompts_dict: Dictionary with sessions, session contexts and prompts.
        """
        self.session_context_prompts_dict = session_context_prompts_dict

    def start(self):
        """
        For each (session, prompt, session_context) tuple, set the context and yield the streamed response.
        """
        for session, context, prompts in self.session_context_prompts_dict:
            print("-> Next session ----------------------------------------")
            session.set_context(context)
            for prompt in prompts:
                print("---> Next prompt -----------------------------------")
                response_stream = session.chat(prompt)
                for response_chunk in response_stream:
                    yield prompt, response_chunk
            print(
                "-> End of session ----------------------------------------", end="\n\n"
            )
