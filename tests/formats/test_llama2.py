import pytest
from llm_templates.common import Content, Conversation
from llm_templates.formats.llama2 import apply_template

def test_llama2_apply_template_basic():
    conversation = Conversation(messages=[
        Content(role="user", content="Hello"),
        Content(role="assistant", content="Hi there!"),
        Content(role="user", content="How are you?"),
    ])
    expected_output = "<s>[INST] Hello [/INST] Hi there! </s><s>[INST] How are you? [/INST]"
    assert apply_template(conversation) == expected_output

def test_llama2_apply_template_with_system_message():
    conversation = Conversation(messages=[
        Content(role="system", content="You are a helpful assistant."),
        Content(role="user", content="What is the weather like?"),
        Content(role="assistant", content="It is sunny."),
        Content(role="user", content="Thank you!"),
    ])
    expected_output = "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nWhat is the weather like? [/INST] It is sunny. </s><s>[INST] Thank you! [/INST]"
    assert apply_template(conversation) == expected_output

def test_llama2_apply_template_add_assistant_prompt():
    conversation = Conversation(messages=[
        Content(role="user", content="Hello"),
    ])
    # The template should add an empty assistant message to the conversation internally
    # and the output should end with the start of an assistant's turn.
    expected_output = "<s>[INST] Hello [/INST] " # Note the space for the expected assistant response start
    formatted_str = apply_template(conversation, add_assistant_prompt=True)
    assert formatted_str == expected_output
    # Check that an empty assistant message was added to the original conversation object
    assert len(conversation.messages) == 2
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == ""

def test_llama2_apply_template_alternate_roles_exception():
    conversation = Conversation(messages=[
        Content(role="user", content="Hello"),
        Content(role="user", content="How are you?"), # Incorrect: user followed by user
    ])
    with pytest.raises(Exception) as excinfo:
        apply_template(conversation)
    assert str(excinfo.value) == "Conversation roles must alternate user/assistant/user/assistant/..."

    conversation_with_system = Conversation(messages=[
        Content(role="system", content="System prompt"),
        Content(role="user", content="Hello"),
        Content(role="user", content="How are you?"), # Incorrect: user followed by user after system
    ])
    with pytest.raises(Exception) as excinfo_system:
        apply_template(conversation_with_system)
    assert str(excinfo_system.value) == "Conversation roles must alternate user/assistant/user/assistant/..."

    conversation_assistant_first = Conversation(messages=[
        Content(role="assistant", content="I'm first?"),
    ])
    with pytest.raises(Exception) as excinfo_assistant_first:
        apply_template(conversation_assistant_first)
    assert str(excinfo_assistant_first.value) == "Conversation roles must alternate user/assistant/user/assistant/..."

    conversation_system_assistant = Conversation(messages=[
        Content(role="system", content="System prompt"),
        Content(role="assistant", content="I'm first after system?"),
    ])
    with pytest.raises(Exception) as excinfo_system_assistant:
        apply_template(conversation_system_assistant)
    assert str(excinfo_system_assistant.value) == "Conversation roles must alternate user/assistant/user/assistant/..."

def test_llama2_apply_template_empty_conversation():
    conversation = Conversation(messages=[])
    expected_output = ""
    assert apply_template(conversation) == expected_output

    # Test with add_assistant_prompt=True as well
    conversation_add_prompt = Conversation(messages=[])
    # When add_assistant_prompt=True on an empty conversation,
    # an assistant prompt (' ') should be generated.
    expected_output_add_prompt = " "
    assert apply_template(conversation_add_prompt, add_assistant_prompt=True) == expected_output_add_prompt
    # Check that one empty assistant message was added
    assert len(conversation_add_prompt.messages) == 1
    assert conversation_add_prompt.messages[0].role == "assistant"
    assert conversation_add_prompt.messages[0].content == ""
