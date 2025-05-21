import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open

from llm_templates.common import Content, Conversation
from llm_templates.formatter import Formatter, save_in_cache, get_from_cache, clear_cache, hf_render

MOCK_USER_HOME_DIR = "/tmp/mock_user_home_for_tests"
DERIVED_CACHE_DIR = os.path.join(MOCK_USER_HOME_DIR, ".cache", "llm_templates")

def create_test_conversation():
    return Conversation(messages=[
        Content(role="user", content="Hello"),
        Content(role="assistant", content="Hi there!"),
        Content(role="user", content="How are you?"),
    ])

@patch('os.path.expanduser', return_value=MOCK_USER_HOME_DIR)
class TestFormatter:

    @patch('llm_templates.formatter.globals') 
    def test_formatter_render_local_model(self, mock_formatter_globals, mock_expanduser_fixture):
        # Pass model_name to constructor
        formatter = Formatter(model_name='llama2') 
        conversation = create_test_conversation()
        expected_result = "formatted_llama2_output"

        mock_apply_template_func = MagicMock(return_value=expected_result)
        mock_llama2_module = MagicMock()
        mock_llama2_module.apply_template = mock_apply_template_func

        mock_formatter_globals.return_value = {'llama2': mock_llama2_module}
        
        with patch('llm_templates.formatter.requests.get') as mock_requests_get:
            mock_requests_get.side_effect = Exception("Network call not expected for local model")
            # Local models don't use Formatter's get_from_cache, so no need to mock it here for this path
            result = formatter.render(conversation) # model_name='llama2' is now in self.model_name

        assert result == expected_result
        # If render is called with no extra kwargs, then _do_render's **kwargs is empty.
        # So, apply_template is called with only conversation.
        mock_apply_template_func.assert_called_once_with(conversation=conversation)
        mock_requests_get.assert_not_called()

    @patch('llm_templates.formatter.requests.get')
    @patch('llm_templates.formatter.save_in_cache') 
    @patch('llm_templates.formatter.hf_render')
    @patch('llm_templates.formatter.get_from_cache', return_value=None) 
    def test_formatter_render_hf_model_success(self, mock_get_from_cache_func, mock_hf_render, mock_save_in_cache_func, mock_requests_get, mock_expanduser_fixture):
        # Pass model_name to constructor
        model_name = 'user/test-model'
        formatter = Formatter(model_name=model_name, huggingface_api_key="test_key") 
        conversation = create_test_conversation()
        expected_config = {"chat_template": "test_template"}
        expected_rendered_string = "rendered_from_hf"

        mock_response_get = MagicMock()
        mock_response_get.status_code = 200
        mock_response_get.json.return_value = expected_config
        mock_requests_get.return_value = mock_response_get
        
        mock_hf_render.return_value = expected_rendered_string
        
        # Pass render-specific kwargs here
        result = formatter.render(conversation, add_assistant_prompt=True, template_name="custom")

        assert result == expected_rendered_string
        expected_url = f"https://huggingface.co/{model_name}/raw/main/tokenizer_config.json"
        mock_requests_get.assert_called_once_with(expected_url, headers={"Authorization": "Bearer test_key"}, allow_redirects=True)
        mock_get_from_cache_func.assert_called_once_with(model_name)
        mock_save_in_cache_func.assert_called_once_with(model_name, expected_config)
        # kwargs from render call are passed through; ensure assertion matches keyword arg passing
        mock_hf_render.assert_called_once_with(conversation=conversation, tokenizer_config=expected_config, add_assistant_prompt=True, template_name="custom")

    @patch('llm_templates.formatter.requests.get')
    @patch('llm_templates.formatter.get_from_cache', return_value=None) 
    def test_formatter_render_hf_model_fetch_failure(self, mock_get_from_cache_func, mock_requests_get, mock_expanduser_fixture):
        model_name = 'user/non-existent-model'
        # Pass model_name to constructor
        formatter = Formatter(model_name=model_name) 
        conversation = create_test_conversation()

        mock_response_get = MagicMock()
        mock_response_get.status_code = 404 
        mock_requests_get.return_value = mock_response_get

        with pytest.raises(ValueError) as excinfo:
            formatter.render(conversation) 
        assert f"Model {model_name} not found in Hugging Face" in str(excinfo.value) # Error from _get_hf_tokenizer
        mock_get_from_cache_func.assert_called_once_with(model_name)

    @patch('llm_templates.formatter.get_from_cache', return_value=None) 
    @patch('llm_templates.formatter.requests.get') 
    @patch('llm_templates.formatter.globals')
    def test_formatter_render_unknown_model_error(self, mock_formatter_globals, mock_requests_get, mock_get_from_cache_func, mock_expanduser_fixture):
        invalid_model_name = "unknown_local_model_without_hf_format"
        # Pass model_name to constructor
        formatter = Formatter(model_name=invalid_model_name) 
        conversation = create_test_conversation()

        mock_formatter_globals.return_value = {} 
        
        with pytest.raises(ValueError) as excinfo:
            formatter.render(conversation)
        
        assert f"Model {invalid_model_name} not found" in str(excinfo.value) # Error from _do_render
        # For this path (unknown, not in globals, not HF format), get_from_cache is NOT called by _do_render
        mock_get_from_cache_func.assert_not_called() 
        mock_requests_get.assert_not_called()

    def test_formatter_render_no_model_error(self, mock_expanduser_fixture):
        formatter = Formatter() # No model_name in constructor
        # Conversation also has no model
        conversation = Conversation(messages=[Content(role="user", content="Hello")]) 
        
        with pytest.raises(ValueError) as excinfo:
            formatter.render(conversation) 
        assert "Model name is required" in str(excinfo.value)

@patch('os.path.expanduser', return_value=MOCK_USER_HOME_DIR)
class TestCachingFunctions:

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_in_cache(self, mock_file_open, mock_path_exists, mock_makedirs, mock_expanduser_fixture_cache):
        model_name = 'test/model'
        config_data = {'key': 'value'}
        
        mock_path_exists.return_value = False
        
        save_in_cache(model_name, config_data) 

        expected_file_path = os.path.join(DERIVED_CACHE_DIR, model_name.replace('/', '_') + "_tokenizer_config.json")
        mock_path_exists.assert_called_once_with(DERIVED_CACHE_DIR) 
        mock_makedirs.assert_called_once_with(DERIVED_CACHE_DIR) 
        mock_file_open.assert_called_once_with(expected_file_path, 'w')
        mock_file_open().write.assert_called_once_with(json.dumps(config_data))

    @patch('os.path.exists', return_value=True) 
    @patch('builtins.open', new_callable=mock_open)
    def test_get_from_cache_exists(self, mock_file_open, mock_path_exists, mock_expanduser_fixture_cache):
        model_name = 'test/model'
        expected_data = {'key': 'value'}
        mock_file_open.return_value.read.return_value = json.dumps(expected_data)
        
        result = get_from_cache(model_name)

        expected_file_path = os.path.join(DERIVED_CACHE_DIR, model_name.replace('/', '_') + "_tokenizer_config.json")
        mock_path_exists.assert_called_once_with(expected_file_path)
        mock_file_open.assert_called_once_with(expected_file_path, 'r')
        assert result == expected_data

    @patch('os.path.exists', return_value=False) 
    def test_get_from_cache_not_exists(self, mock_path_exists, mock_expanduser_fixture_cache):
        model_name = 'test/nonexistent-model'
        
        result = get_from_cache(model_name)

        expected_file_path = os.path.join(DERIVED_CACHE_DIR, model_name.replace('/', '_') + "_tokenizer_config.json")
        mock_path_exists.assert_called_once_with(expected_file_path)
        assert result is None

    @patch('os.unlink')
    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.exists') 
    @patch('os.path.join', new_callable=lambda: (lambda *args: os.path.normpath(os.sep.join(args))))
    def test_clear_cache(self, mock_os_join_lambda, mock_path_exists_cache_dir, mock_listdir, mock_isfile, mock_unlink, mock_expanduser_fixture_cache):
        mock_path_exists_cache_dir.return_value = True
        
        mock_files = ["file1.json", "file2.txt", "file3.json"]
        mock_listdir.return_value = mock_files
        
        mock_isfile.return_value = True

        clear_cache()

        mock_path_exists_cache_dir.assert_called_once_with(DERIVED_CACHE_DIR)
        mock_listdir.assert_called_once_with(DERIVED_CACHE_DIR)
        
        assert mock_unlink.call_count == len(mock_files)
        for f_name in mock_files:
            expected_path = os.path.normpath(os.sep.join([DERIVED_CACHE_DIR, f_name]))
            mock_isfile.assert_any_call(expected_path)
            mock_unlink.assert_any_call(expected_path)

class TestHfRender:
    def test_hf_render_basic(self):
        conversation = create_test_conversation()
        tokenizer_config = {
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}"
        }
        expected_output = "USER: Hello\nASSISTANT: Hi there!\nUSER: How are you?\n"
        result = hf_render(conversation, tokenizer_config)
        assert result == expected_output

    def test_hf_render_with_add_generation_prompt(self):
        conversation = Conversation(messages=[Content(role="user", content="Hello")])
        tokenizer_config = {
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        }
        expected_output_prompt = "USER: Hello\nASSISTANT:"
        result_prompt = hf_render(conversation, tokenizer_config, add_assistant_prompt=True)
        assert result_prompt == expected_output_prompt
        assert len(conversation.messages) == 1 

        expected_output_no_prompt = "USER: Hello\n"
        result_no_prompt = hf_render(conversation, tokenizer_config, add_assistant_prompt=False)
        assert result_no_prompt == expected_output_no_prompt
    
    def test_hf_render_with_list_of_templates(self):
        conversation = create_test_conversation()
        tokenizer_config = {
            "chat_template": [
                {"name": "default", "template": "default_template_output"},
                {"name": "specific_template_name", "template": "{% for message in messages %}{{ message['role'] }}: {{ message['content']}} {% endfor %}"}
            ]
        }
        expected_output = "user: Hello assistant: Hi there! user: How are you? "
        result = hf_render(conversation, tokenizer_config, template_name="specific_template_name")
        assert result == expected_output

    def test_hf_render_list_template_not_found(self):
        conversation = create_test_conversation()
        tokenizer_config = { "chat_template": [ {"name": "default", "template": "default_template_output"}, ] }
        with pytest.raises(ValueError, match="Template 'non_existent_template' not found in tokenizer_config"):
            hf_render(conversation, tokenizer_config, template_name="non_existent_template")

    def test_hf_render_no_chat_template(self):
        conversation = create_test_conversation()
        tokenizer_config = {} 
        with pytest.raises(ValueError, match="No chat_template found in tokenizer_config"):
            hf_render(conversation, tokenizer_config)

    def test_hf_render_invalid_chat_template_type(self):
        conversation = create_test_conversation()
        tokenizer_config = {"chat_template": 123} 
        with pytest.raises(ValueError, match="Invalid chat_template format in tokenizer_config"):
            hf_render(conversation, tokenizer_config)

    def test_hf_render_empty_conversation(self):
        conversation = Conversation(messages=[])
        tokenizer_config = {
            "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content']}}{% endfor %}{% if add_generation_prompt %}ASSISTANT:{% endif %}"
        }
        expected_output_empty = ""
        result_empty = hf_render(conversation, tokenizer_config, add_assistant_prompt=False)
        assert result_empty == expected_output_empty

        expected_output_prompt = "ASSISTANT:"
        result_prompt = hf_render(conversation, tokenizer_config, add_assistant_prompt=True)
        assert result_prompt == expected_output_prompt
        assert len(conversation.messages) == 0
