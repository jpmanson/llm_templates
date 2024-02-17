from llm_templates.common import Conversation, Content, get_jinja_env


def apply_template(conversation: Conversation, **kwargs):
    if kwargs.get('add_assistant_prompt', False):
        conversation.messages.append(Content(role='assistant', content=''))

    # Definir la plantilla Jinja
    bos_token = "<s>"
    eos_token = "</s>"
    template_str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

    # Crear un objeto de plantilla usando el entorno con el filtro personalizado
    template = get_jinja_env().from_string(template_str)

    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(messages=conversation.messages, bos_token=bos_token, eos_token=eos_token)