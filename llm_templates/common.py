from pydantic import BaseModel
from typing import List, Optional
from jinja2 import Environment


class Content(BaseModel):
    role: str
    content: str


# simplified version of openai.types.chat completion_create_params.py
class Conversation(BaseModel):
    model: Optional[str]
    messages: List[Content]

    def __str__(self):
        return f"model: {self.model}, messages: {self.messages}"

    def __repr__(self):
        return f"model: {self.model}, messages: {self.messages}"

    def append_prompt(self, role: str = None, message: str = None, content: Content = None):
        if content is not None:
            self.messages.append(content)
        else:
            self.messages.append(Content(role=role, content=message))
        return self


# Function to raise exception from Jinja2 templates
def raise_exception(message):
    raise Exception(message)


def get_jinja_env(**kwargs):
    jinja_env = Environment()
    jinja_env.globals['raise_exception'] = raise_exception
    jinja_env.globals.update(kwargs)
    return jinja_env

