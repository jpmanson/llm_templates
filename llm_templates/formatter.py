from llm_templates.common import Conversation, get_jinja_env
import requests
import os
import json
from llm_templates.formats import zephyr, mistral, llama2, gemma, llama3, cohere, phi3, llama31

HF_URL = 'https://huggingface.co/'


class Formatter:
    def __init__(self, model_name: str = None, use_cache: bool = True,
                 clear_cache_on_create: bool = False, huggingface_api_key: str = None):
        self.model_name = model_name
        self.use_cache = use_cache
        self.huggingface_api_key = huggingface_api_key
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
            headers = {}
            if self.huggingface_api_key:
                headers["Authorization"] = f"Bearer {self.huggingface_api_key}"
            url = f"{HF_URL}{model}/raw/main/tokenizer_config.json"
            response = requests.get(url, headers=headers, allow_redirects=True)
            if response.status_code == 200:
                tokenizer_config = response.json()
                save_in_cache(model, tokenizer_config)
            elif response.status_code == 404:
                raise ValueError(f"Model {model} not found in Hugging Face")
            elif response.status_code == 403:
                raise ValueError(f"Model {model} forbidden. "
                                 f"Some models require a token: Formatter(huggingface_api_key=HUGGINGFACE_TOKEN) and agreement with the model's license")
            else:
                raise ValueError(f"Model {model} failed to load. Status code: {response.status_code}")

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

    template_name = kwargs.get("template_name", 'default')

    template_str = tokenizer_config.get("chat_template", '')

    if isinstance(template_str, list):
        template_str = [t for t in template_str if t.get('name') == template_name][0]['template']
    elif isinstance(template_str, dict):
        template_str = template_str.get('template')

    template = get_jinja_env().from_string(template_str)

    for k, v in tokenizer_config.items():
        if isinstance(v, dict) and tokenizer_config[k].get('__type') == 'AddedToken':
            tokenizer_config[k] = str(tokenizer_config[k].get('content'))

    return template.render(messages=conversation.messages, **tokenizer_config,
                           add_generation_prompt=add_generation_prompt)
