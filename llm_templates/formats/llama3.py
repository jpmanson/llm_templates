from llm_templates.common import Conversation, get_jinja_env
from jinja2 import Environment

# Source format: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/


def apply_template(conversation: Conversation, **kwargs):
    template_str = "{{bos_token}}{% for message in messages %}"
    template_str += "{% if message['role'] == 'user' %}"
    template_str += "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>\n"
    template_str += "{% elif message['role'] == 'assistant' %}"
    template_str += "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>\n"
    template_str += "{% elif message['role'] == 'system' %}"
    template_str += "<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>\n"
    template_str += "{% else %}"
    template_str += "<|start_header_id|>unknown<|end_header_id|>\n{{ message['content'] }}<|eot_id|>\n"
    template_str += "{% endif %}"
    template_str += "{% endfor %}"
    template_str += "{% if add_generation_prompt %}"
    template_str += "<|start_header_id|>assistant<|end_header_id|>\n"
    template_str += "{% endif %}"

    # Load template
    bos_token = "<|begin_of_text|>"
    eos_token = "<|end_of_text|>"
    template = get_jinja_env().from_string(template_str)

    # Renderize the template with the provided messages
    return template.render(messages=conversation.messages, bos_token=bos_token, eos_token=eos_token,
                           add_generation_prompt=kwargs.get('add_assistant_prompt', False))
