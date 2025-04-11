import os
from mistralai import Mistral
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ✅ Extract profile context from MongoDB
def get_profiles_context():
    try:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            return "MongoDB URI not set in environment."

        client = MongoClient(mongo_uri)
        db = client["app-dev"]
        collection = db["profiles"]
        profiles = list(collection.find({}, {"_id": 0}))

        if not profiles:
            return "No profiles found."

        context_parts = []
        for profile in profiles:
            skills = [s.get("skill", "") for s in profile.get("highlightedSkills", [])]
            context_parts.append(f"""
    Name: {profile.get('firstName', '')} {profile.get('lastName', '')}
    Expertise: {profile.get('areaOfExpertise', '')}
    Summary: {profile.get('carrierSummary', '')}
    Skills: {', '.join(skills)}
    Experience: {profile.get('experience', [])}
            """)

        return "\n".join(context_parts)

    except Exception as e:
        return f"Error retrieving profiles: {e}"

class ChatBot:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.mistral_client = Mistral(api_key=api_key)
        self.initialize_context()

    def initialize_context(self):
        context = get_profiles_context()
        system_message = {
            "role": "system",
            "content": f"You are a helpful assistant who answers questions based on the following user profiles:\n\n{context}"
        }
        self.conversation_history.append(system_message)

    def run(self):
        while True:
            self.get_user_input()
            self.send_request()

    def get_user_input(self):
        user_input = input("\nYou: ")
        user_message = {
            "role": "user",
            "content": user_input
        }
        self.conversation_history.append(user_message)
        return user_message

    def send_request(self):
        buffer = ""
        stream_response = self.mistral_client.chat.stream(
            model=self.model,
            messages=self.conversation_history
        )
        for chunk in stream_response:
            content_part = chunk.data.choices[0].delta.content
            if content_part:
                buffer += content_part
                print(content_part, end="", flush=True)

        print("\n")
        assistant_message = {
            "role": "assistant",
            "content": buffer
        }
        self.conversation_history.append(assistant_message)

if __name__ == "__main__":
    api_key = os.getenv('MISTRAL_API_KEY')
    model = "mistral-large-latest"
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    chat_bot = ChatBot(api_key, model)
    chat_bot.run()
