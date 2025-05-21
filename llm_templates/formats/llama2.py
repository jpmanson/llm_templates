from llm_templates.common import Conversation, Content, get_jinja_env

jinja_tpl = """\
{%- if not messages -%}
    {{- '' -}}
{%- else -%}
    {%- if messages[0]['role'] == 'system' -%}
        {%- set loop_messages = messages[1:] -%}
        {%- set system_message = '<<SYS>>\\n' + messages[0]['content'].strip() + '\\n<</SYS>>\\n\\n' -%}
    {%- else -%}
        {%- set loop_messages = messages -%}
        {%- set system_message = '' -%}
    {%- endif -%}
    {%- for message in loop_messages -%}
        {# Role alternation check:
           - If there's a system message, the first message in loop_messages (index 0) must be 'user'.
           - If no system message, the first message in messages (index 0) must be 'user'.
           - Subsequent messages must alternate.
           - Exception: if add_assistant_prompt_flag is true AND it's the only message, it can be an assistant message.
        #}
        {%- if not (loop.first and message['role'] == 'assistant' and loop_messages|length == 1 and add_assistant_prompt_flag) -%}
            {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
                {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
            {%- endif -%}
        {%- endif -%}
        {%- if loop.index0 == 0 and not (message['role'] == 'assistant' and add_assistant_prompt_flag) %}
            {%- set content = system_message + message['content'] -%}
        {%- else -%}
            {%- set content = message['content'] -%}
        {%- endif -%}
        {%- if message['role'] == 'user' -%}
            {{- bos_token + '[INST] ' + content.strip() + ' [/INST]' -}}
        {%- elif message['role'] == 'assistant' -%}
            {%- set stripped_content = content.strip() -%}
            {%- if loop.last and add_assistant_prompt_flag -%}
                {# If it's the last message and an auto-prompt, output just a space if content is empty, else content + space #}
                {%- if stripped_content == '' -%}
                    {{- ' ' -}}
                {%- else -%}
                    {{- ' ' + stripped_content + ' ' -}}
                {%- endif -%}
            {%- else -%}
                {{- ' ' + stripped_content + ' ' + eos_token -}}
            {%- endif -%}
        {%- endif -%}
    {%- endfor -%}
{%- endif -%}\
"""


def apply_template(conversation: Conversation, **kwargs):
    add_assistant_prompt = kwargs.get('add_assistant_prompt', False)
    # Only add an empty assistant message if add_assistant_prompt is true AND
    # (the conversation is empty OR the last message is from a user).
    if add_assistant_prompt and (not conversation.messages or conversation.messages[-1].role == "user"):
        # Ensure we don't add multiple assistant prompts if called multiple times.
        if not conversation.messages or conversation.messages[-1].role != 'assistant' or conversation.messages[-1].content != '':
             conversation.messages.append(Content(role='assistant', content=''))

    # Load template
    bos_token = "<s>"
    eos_token = "</s>"
    # Pass add_assistant_prompt_flag to the template context
    template = get_jinja_env().from_string(jinja_tpl)

    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(
        messages=conversation.messages,
        bos_token=bos_token,
        eos_token=eos_token,
        add_assistant_prompt_flag=add_assistant_prompt  # Pass the flag to the template
    )
