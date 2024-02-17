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

# You can add your hf key with Formatter(huggingface_api_key="your_key").
# Some models require authentication (For example: "meta-llama/Llama-2-7b-chat-hf")
formatter = Formatter()

# Hugging Face models
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model = "HuggingFaceH4/zephyr-7b-beta"
# model = "meta-llama/Llama-2-7b-chat-hf"
conversation = Conversation(model=model, messages=messages)
conversation_str = formatter.render(conversation, add_assistant_prompt=True)

from huggingface_hub import InferenceClient
client = InferenceClient()  # client = InferenceClient(token="hf_***") # If you have a token

result = client.text_generation(prompt=conversation_str, model=model, max_new_tokens=768, temperature=0.7, top_p=0.9,
                                top_k=50)

print(result)
