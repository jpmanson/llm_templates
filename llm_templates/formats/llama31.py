from llm_templates.common import Conversation, get_jinja_env

# Source format: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/raw/main/tokenizer_config.json


template_str = """
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}
<|start_header_id|>system<|end_header_id|>
{%- if builtin_tools is defined or tools is not none %}
Environment: ipython
{%- endif %}
{%- if builtin_tools is defined %}
Tools: {{ builtin_tools | reject('equalto', 'code_interpreter') | join(", ") }}
{%- endif %}
Cutting Knowledge Date: December 2023
Today Date: {{ date_string }}
{%- if tools is not none and not tools_in_user_message %}
You have access to the following functions. To call a function, please respond with JSON for a function call.
Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.
Do not use variables.

{%- for t in tools %}
    {{ t | tojson(indent=4) }}
{%- endfor %}
{%- endif %}

{{ system_message }}<|eot_id|>

{%- if tools_in_user_message and not tools is none %}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{ raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
<|start_header_id|>user<|end_header_id|>
Given the following functions, please respond with a JSON for a function call 
with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.
Do not use variables.

{%- for t in tools %}
    {{ t | tojson(indent=4) }}
{%- endfor %}

{{ first_user_message }}<|eot_id|>
{%- endif %}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] | trim }}<|eot_id|>
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{ raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
<|start_header_id|>assistant<|end_header_id|>
<|python_tag|>{{ tool_call.name }}.call(
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{ arg_name }}="{{ arg_val }}"
                {%- if not loop.last %}, {% endif %}
            {%- endfor %}
)
        {%- else %}
<|start_header_id|>assistant<|end_header_id|>
{"name": "{{ tool_call.name }}", "parameters": {{ tool_call.arguments | tojson }}}
        {%- endif %}
        {%- if builtin_tools is defined %}
<|eom_id|>
        {%- else %}
<|eot_id|>
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
<|start_header_id|>ipython<|end_header_id|>
        {%- if message.content is mapping or message.content is iterable %}
{{ message.content | tojson }}
        {%- else %}
{{ message.content }}
        {%- endif %}
<|eot_id|>
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""


def apply_template(conversation: Conversation, **kwargs):
    # Load template
    bos_token = "<|begin_of_text|>"
    eos_token = "<|end_of_text|>"
    template = get_jinja_env().from_string(template_str)

    # Renderize the template with the provided messages
    return template.render(messages=conversation.messages, bos_token=bos_token, eos_token=eos_token,
                           add_generation_prompt=kwargs.get('add_assistant_prompt', False),
                           **kwargs)
