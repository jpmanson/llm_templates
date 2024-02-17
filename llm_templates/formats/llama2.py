from llm_templates.common import Conversation, Content, get_jinja_env

jinja_tpl = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = '<<SYS>>\n' + messages[0]['content'].strip() + '\n<</SYS>>\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


def apply_template(conversation: Conversation, **kwargs):
    if kwargs.get('add_assistant_prompt', False):
        conversation.messages.append(Content(role='assistant', content=''))

    # Load template
    bos_token = "<s>"
    eos_token = "</s>"
    template = get_jinja_env().from_string(jinja_tpl)

    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(messages=conversation.messages, bos_token=bos_token, eos_token=eos_token)
