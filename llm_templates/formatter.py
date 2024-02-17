from llm_templates.common import Conversation, jinja_env
import requests
import os
import json
from llm_templates.formats import zephyr, mistral

HF_URL = 'https://huggingface.co/'


class Formatter:
    def __init__(self, model_name: str = None, use_cache: bool = True, clear_cache_on_create: bool = False):
        self.model_name = model_name
        self.use_cache = use_cache
        if clear_cache_on_create:
            clear_cache()

    def render(self, conversation: Conversation, **kwargs) -> str:
        model = self.model_name or conversation.model
        if model:
            return self._do_render(model, conversation, **kwargs)

        else:
            raise ValueError("Model name is required")

    def _get_hf_tokenizer(self, model: str) -> dict:
        tokenizer_config = None
        if self.use_cache:
            tokenizer_config = get_from_cache(model)

        if not tokenizer_config:
            # Get model tokenizer json
            url = f"{HF_URL}{model}/raw/main/tokenizer_config.json"
            response = requests.get(url)
            if response.status_code == 200:
                tokenizer_config = response.json()
                save_in_cache(model, tokenizer_config)
            else:
                raise ValueError(f"Model {model} configuration not found in server")

        return tokenizer_config

    def _do_render(self, model: str, conversation: Conversation, **kwargs) -> str:
        if model not in globals():
            # Check if is in HuggingFace format: "user/model"
            is_hugging_face_fmt = model.count("/") == 1
            if is_hugging_face_fmt:
                tokenizer_config = self._get_hf_tokenizer(model)

                return hf_render(conversation=conversation, tokenizer_config=tokenizer_config, **kwargs)
            else:
                raise ValueError(f"Model {model} not found")

        template_module = globals()[model]
        return template_module.apply_template(conversation=conversation, **kwargs)


def save_in_cache(model: str, tokenizer_config: dict) -> None:
    os_home_folder = os.path.expanduser("~")
    file_name = model.replace("/", "_") + "_tokenizer_config.json"
    file_path = f"{os_home_folder}/.cache/llm_templates/{file_name}"

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, "w") as f:
        f.write(json.dumps(tokenizer_config))


def get_from_cache(model: str) -> dict | None:
    os_home_folder = os.path.expanduser("~")
    file_name = model.replace("/", "_") + "_tokenizer_config.json"
    file_path = f"{os_home_folder}/.cache/llm_templates/{file_name}"

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        return json.load(f)


def clear_cache() -> None:
    os_home_folder = os.path.expanduser("~")
    cache_folder = f"{os_home_folder}/.cache/llm_templates"
    if os.path.exists(cache_folder):
        for file in os.listdir(cache_folder):
            file_path = os.path.join(cache_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


def hf_render(conversation: Conversation, tokenizer_config: dict, **kwargs) -> str:
    add_generation_prompt = kwargs.get("add_assistant_prompt", False)

    template_str = tokenizer_config.get("chat_template", '')

    template = jinja_env.from_string(template_str)

    return template.render(messages=conversation.messages, **tokenizer_config,
                           add_generation_prompt=add_generation_prompt)
