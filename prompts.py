from pydantic import BaseModel

from .base import BasePrompt


class ActionLabel(BaseModel):
    description: str
    action: str


SINGLE_IMAGE_PROMPT = """Classify the action of the person in the **purple mask**. \
For object interactions the object must be visible. If the highlighted person is not \
performing any action or is not a real person, choose "None".

Choose from the list:

{actions}

Only respond with a JSON object with the following format:
{{
    "action": "..."
    "description": "..."
}}
"""


class HumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        return SINGLE_IMAGE_PROMPT.format(
            actions="\n".join(f'- "{action}"' for action in self.actions),
            schema=ActionLabel.model_json_schema(),
            example=ActionLabel(
                description="This is a description", action="walking"
            ).model_dump_json(),
        ).strip("\n")


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
    "action": "..."
    "description": "..."
}}
"""


class MultiImageHumanActionPrompt(BasePrompt):
    actions: list[str]

    def __init__(self, actions: list[str]):
        self.actions = actions

    def prompt(self) -> str:
        return (
            MULTI_IMAGE_PROMPT.format(
                actions="\n".join(f'- "{action}"' for action in self.actions),
                schema=ActionLabel.model_json_schema(),
                example=ActionLabel(
                    description="This is a description", action="walking"
                ).model_dump_json(),
            )
            .strip()
            .strip("\n")
        )
