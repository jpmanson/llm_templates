from pydantic import BaseModel
from typing import List, Optional
from jinja2 import Environment


class Content(BaseModel):
    role: str
    content: str


# simplified version of openai.types.chat completion_create_params.py
class Conversation(BaseModel):
    model: Optional[str] = None
    messages: List[Content]

    def __str__(self):
        return f"model: {self.model}, messages: {self.messages}"

    def __repr__(self):
        return f"model: {self.model}, messages: {self.messages}"

    def append_prompt(self, role_or_content: Optional[str | Content] = None, message: Optional[str] = None):
        if isinstance(role_or_content, Content):
            self.messages.append(role_or_content)
        elif role_or_content is not None and message is not None:
            self.messages.append(Content(role=role_or_content, content=message))
        else:
            # Or raise an error if arguments are not as expected
            raise ValueError("append_prompt requires either a Content object or role and message strings.")
        return self


# Function to raise exception from Jinja2 templates
def raise_exception(message):
    raise Exception(message)


def get_jinja_env(**kwargs):
    jinja_env = Environment()
    jinja_env.globals['raise_exception'] = raise_exception
    jinja_env.globals.update(kwargs)
    return jinja_env

