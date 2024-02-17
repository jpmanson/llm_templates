from llm_templates.common import Conversation, get_jinja_env
from jinja2 import Environment


def apply_template(conversation: Conversation, **kwargs):
    template_str = "{% for message in messages %}"
    template_str += "{% if message['role'] == 'user' %}"
    template_str += "<|user|>{{ message['content'] }}</s>\n"
    template_str += "{% elif message['role'] == 'assistant' %}"
    template_str += "<|assistant|>{{ message['content'] }}</s>\n"
    template_str += "{% elif message['role'] == 'system' %}"
    template_str += "<|system|>{{ message['content'] }}</s>\n"
    template_str += "{% else %}"
    template_str += "<|unknown|>{{ message['content'] }}</s>\n"
    template_str += "{% endif %}"
    template_str += "{% endfor %}"
    template_str += "{% if add_generation_prompt %}"
    template_str += "<|assistant|>\n"
    template_str += "{% endif %}"

    # Load template
    bos_token = "<s>"
    eos_token = "</s>"
    template = get_jinja_env().from_string(template_str)

    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(messages=conversation.messages, bos_token=bos_token, eos_token=eos_token,
                           add_generation_prompt=kwargs.get('add_assistant_prompt', False))
