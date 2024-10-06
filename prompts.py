from .base import BasePrompt


class HumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        # prompt = "What is the person in the purple mask currently doing? \n\nChoose from the list:\n\n"
        prompt = "Classify the person in the purple box. For object interactions the object must be visible. Choose from the list:\n\n"

        for action in self.actions:
            prompt += f"- {action}\n"

        prompt += f"- None of the above\n\n"

        prompt += 'Respond with a valid JSON object:\n{\n    "Description": "describe the person",\n    "Action": "chosen action",\n    "Confidence": confidence score (0 to 1)\n}'

        return prompt


MULTI_IMAGE_PROMPT = """
You will be given 3 images, each with a different level of cropping.
The first image is not cropped at all to give full context.
The second image is cropped to show the person to classify with a purple box and some of the background.
The third image is cropped to show the person to classify.

Classify the action of the person in the **purple box**. \
For object interactions the object must be visible.

Choose from the list:

{actions}

Respond with a JSON object with the following format:
{{
    "Description": describe the person you're classifying
    "Action": the action the person is performing
    "Confidence": confidence in the prediction
}}
"""


class MultiImageHumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        return (
            MULTI_IMAGE_PROMPT.format(
                actions="\n".join(f"- {action}" for action in self.actions)
            )
            .strip()
            .strip("\n")
        )
