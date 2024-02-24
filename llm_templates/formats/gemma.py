from llm_templates.common import Conversation, Content, get_jinja_env


# Source format: https://ai.google.dev/gemma/docs/formatting?hl=es-419


def apply_template(conversation: Conversation, **kwargs):
    # Definir la plantilla Jinja
    template_str = "{% for message in messages %}"
    template_str += "{% if message['role'] == 'user' %}"
    template_str += "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    template_str += "{% elif message['role'] == 'assistant' %}"
    template_str += "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    template_str += "{% elif message['role'] == 'system' %}"
    template_str += "<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
    template_str += "{% else %}"
    template_str += "<start_of_turn>unknown\n{{ message['content'] }}<end_of_turn>\n"
    template_str += "{% endif %}"
    template_str += "{% endfor %}"
    template_str += "{% if add_generation_prompt %}"
    template_str += "<start_of_turn>model\n"
    template_str += "{% endif %}"

    template = get_jinja_env().from_string(template_str)

    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(messages=conversation.messages,
                           add_generation_prompt=kwargs.get('add_assistant_prompt', False))