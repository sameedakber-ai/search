from django_unicorn.components import UnicornView


class LlmView(UnicornView):
    question = "How do I do this?"

    def add_to_llm_conversation(self):
        print(self.question)

