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


# Function to raise exception from Jinja2 templates
def raise_exception(message):
    raise Exception(message)


jinja_env = Environment()
jinja_env.globals['raise_exception'] = raise_exception
