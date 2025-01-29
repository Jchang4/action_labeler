from abc import ABC, abstractmethod

import supervision as sv
from PIL import Image

from action_labeler.v2.base import IPrompt


class BaseActionPrompt(IPrompt):
    template: str
    actions: list[str]
    numbered_actions: bool = False

    def __init__(
        self, template: str, actions: list[str], numbered_actions: bool = False
    ):
        self.template = template
        self.actions = actions
        self.numbered_actions = numbered_actions

    def format_actions(self) -> str:
        if self.numbered_actions:
            return "\n".join(
                f'{i+1}. "{action}"' for i, action in enumerate(self.actions)
            )
        return "\n".join(f'- "{action}"' for action in self.actions)
