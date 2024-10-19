from .base import BasePrompt

SINGLE_IMAGE_PROMPT = """Classify the action of the person in the **purple mask**. \
For object interactions the object must be visible. If the highlighted person is not \
performing any action or is not a real person, choose "None".

Choose from the list:

{actions}

Only respond with a JSON object with the following format:
{{
    "Description": "string: describe the person you're classifying"
    "Action": "string: the action the person is performing, must be exact text from the list"
    "Confidence": "string: confidence in the prediction from 0 to 1."
}}"""


class HumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        return (
            SINGLE_IMAGE_PROMPT.format(
                actions="\n".join(f'- "{action}"' for action in self.actions)
            )
            .strip()
            .strip("\n")
        )


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
    "Action": the action the person is performing, must be exact text from the list
    "Confidence": confidence in the prediction from 0 to 1.
}}
"""


class MultiImageHumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        return (
            MULTI_IMAGE_PROMPT.format(
                actions="\n".join(f'- "{action}"' for action in self.actions)
            )
            .strip()
            .strip("\n")
        )
