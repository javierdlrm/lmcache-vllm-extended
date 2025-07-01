import random


class RequestGenerator:
    def __init__(self, system_prompt, session_context_prompts_dict):
        """
        :param session_context_prompts_dict: Dictionary with sessions, session contexts and prompts.
        """
        self.system_prompt = system_prompt
        self.session_context_prompts_dict = session_context_prompts_dict

    def start(self, randomize=True):
        """
        For each (session, context, prompts) dict, set the context and yield the streamed response.
        """

        # All session and prompts combinations
        session_prompt_tuples = []

        for entry in self.session_context_prompts_dict.values():
            session = entry["session"]
            context = entry["context"]
            prompts = entry["prompts"]

            extended_context = [context]
            session.set_context(extended_context)

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

        # for entry in self.session_context_prompts_dict.values():
        #     session = entry["session"]
        #     context = entry["context"]
        #     prompts = entry["prompts"]

        #     # extended_context = [self.system_prompt] + [context]
        #     extended_context = [context]

        #     session.set_context(extended_context)

        #     for prompt in prompts:
        #         response_stream = session.chat(prompt)
        #         for response_chunk in response_stream:
        #             yield prompt, response_chunk
