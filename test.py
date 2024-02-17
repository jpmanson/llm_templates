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

# Local model
conversation = Conversation(model="llama2", messages=messages)
print(formatter.render(conversation))

# Hugging Face models
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model = "HuggingFaceH4/zephyr-7b-beta"
conversation = Conversation(model=model, messages=messages)
conversation_str = formatter.render(conversation, add_assistant_prompt=True)

from huggingface_hub import InferenceClient
client = InferenceClient()

result = client.text_generation(prompt=conversation_str, model=model, max_new_tokens=768, temperature=0.7, top_p=0.9,
                                top_k=50)

print(result)
