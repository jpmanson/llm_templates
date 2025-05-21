import pytest
from llm_templates.common import Content, Conversation

def test_content_creation():
    role = "user"
    content_text = "Hello, world!"
    content_obj = Content(role=role, content=content_text)
    assert content_obj.role == role
    assert content_obj.content == content_text

def test_conversation_creation():
    messages = [
        Content(role="user", content="Hello"),
        Content(role="assistant", content="Hi there!")
    ]
    model_name = "test-model"
    conversation = Conversation(model=model_name, messages=messages)
    assert conversation.model == model_name
    assert conversation.messages == messages

    # Test creation without a model name
    conversation_no_model = Conversation(messages=messages)
    assert conversation_no_model.model is None
    assert conversation_no_model.messages == messages

def test_conversation_append_prompt():
    initial_messages = [Content(role="user", content="Initial prompt")]
    conversation = Conversation(model="test-model", messages=initial_messages)

    # Test appending with role and message string
    appended_conversation = conversation.append_prompt("assistant", "Response 1")
    assert len(conversation.messages) == 2
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == "Response 1"
    assert appended_conversation is conversation # Check for chaining

    # Test appending with a Content object
    new_content = Content(role="user", content="Another prompt")
    appended_conversation = conversation.append_prompt(new_content)
    assert len(conversation.messages) == 3
    assert conversation.messages[2] == new_content
    assert appended_conversation is conversation # Check for chaining
