from llm_templates.common import Conversation, get_jinja_env
from jinja2 import Environment


def apply_template(conversation: Conversation, **kwargs):
    template_str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"

    # Load template
    bos_token = "<s>"
    eos_token = "<|endoftext|>"
    template = get_jinja_env().from_string(template_str)

    # Renderize the template with the provided messages
    return template.render(messages=conversation.messages, bos_token=bos_token, eos_token=eos_token,
                           add_generation_prompt=kwargs.get('add_assistant_prompt', False))
