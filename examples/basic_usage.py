from llm_templates import Formatter, Conversation

messages = [
    {
        "role": "user",
        "content": "Hello!"
    },
    {
        "role": "assistant",
        "content": "How can I help you?"
    },
    {
        "role": "user",
        "content": "Write a poem about the sea"
    }
]

formatter = Formatter()

# Local model zephyr
conversation = Conversation(model="zephyr", messages=messages)
print(formatter.render(conversation))

# Local model mistral
conversation = Conversation(model="mistral", messages=messages)
print(formatter.render(conversation))