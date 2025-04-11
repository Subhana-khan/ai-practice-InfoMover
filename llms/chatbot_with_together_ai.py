
# Just a small description regarding Together.ai :-
#        Together.ai is a platform that hosts and supports various top open-source language models —
#         like Meta('s LLaMA, Mistral, and Mixtral. I can access these models directly through Together’s unified API, '
#         without needing to install or manage separate SDKs for each model. That made it super convenient to integrate any
#         of them into my chatbot just by changing the model name in one line of code.)

import os
import together

# Load environment variables (optional, if using a .env file)
# Note: Uncomment the next line if you have python-dotenv installed and a .env file
from dotenv import load_dotenv
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("TOGETHER_API_KEY")

# Check if API key is set
if api_key is None:
    print("You need to set your TOGETHER_API_KEY environment variable!")
    exit(1)

# Initialize the Together client
together.api_key = api_key
client = together.Together()

# ("I used Together.ai to access Meta’s LLaMA 3.3 (70B Instruct Turbo) model, which is a powerful open-weight LLM."
#  " Together.ai made it easy to use the model via API without needing to host it myself.")

model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
messages = [
    {
        "role": "user",
        "content": "Where is the Oxford University located ??"
    }
]

# Get the response from the model
chat_response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=150
)

# Print the response
print(chat_response.choices[0].message.content)