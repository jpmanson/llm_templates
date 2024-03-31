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