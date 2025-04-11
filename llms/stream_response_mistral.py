import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)

    # Stream the chat response
    stream_response = client.chat.stream(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "list 10 hospitals of bhopal?",
            },
        ]
    )

    # Iterate over the stream and print each chunk
for chunk in stream_response:
    print(chunk.data.choices[0].delta.content, end="")

