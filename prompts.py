from .base import BasePrompt


class HumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        # prompt = "What is the person in the purple mask currently doing? \n\nChoose from the list:\n\n"
        prompt = "Classify the person with the purple filter. For object interactions the object must be visible. Choose from the list:\n\n"

        for action in self.actions:
            prompt += f"- {action}\n"

        prompt += f"- None of the above\n\n"

        prompt += 'Respond with a valid JSON object:\n{\n    "Description": "describe the person",\n    "Action": "chosen action",\n    "Confidence": confidence score (0 to 1)\n}'

        return prompt
