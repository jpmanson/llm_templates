from llm_templates import Formatter, Conversation, Content

messages = [Content(role="user", content="Hello!"),
            Content(role="assistant", content="How can I help you?"),
            Content(role="user", content="Write a poem about the sea")]

formatter = Formatter()

# Local model mistral
print('\n# mistral')
conversation = Conversation(model="mistral", messages=messages)
print(formatter.render(conversation))

# Local model llama2
print('\n# llama2')
conversation = Conversation(model="llama2", messages=messages)
print(formatter.render(conversation))

# Local model zephyr
print('\n# zephyr')
conversation = Conversation(model="zephyr", messages=messages)
print(formatter.render(conversation))

# Local model gemma
print('\n# gemma')
conversation = Conversation(model="gemma", messages=messages)
print(formatter.render(conversation))

# Local model llama3
print('\n# llama3')
conversation = Conversation(model="llama3", messages=messages)
print(formatter.render(conversation, add_assistant_prompt=True))

# Local model cohere default chat template
print('\n# cohere')
conversation = Conversation(model="cohere", messages=messages)
print(formatter.render(conversation, add_assistant_prompt=True))

# Local model phi3
print('\n# phi3')
conversation = Conversation(model="phi3", messages=messages)
print(formatter.render(conversation, add_assistant_prompt=True))

# Local model llama 3.1
print('\n# llama 3.1')
conversation = Conversation(model="llama31", messages=messages)
print(formatter.render(conversation, add_assistant_prompt=True))