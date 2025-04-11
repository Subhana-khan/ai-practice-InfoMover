
import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ChatMessageHistory
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not set in environment variables. Please check your .env file.")

try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')  # Test connection
    print("✅ Successfully connected to MongoDB!")
except (ConnectionFailure, ServerSelectionTimeoutError, ConfigurationError) as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# Access database and collection
db = client["app-dev"]
profiles_collection = db["profiles"]

# Check if collection exists and has documents
if "profiles" not in db.list_collection_names():
    print("⚠️ Warning: 'profiles' collection not found.")
else:
    count = profiles_collection.count_documents({})
    print(f"📁 'profiles' collection found with {count} documents.")


def profile_to_context_string(profile):
    full_name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
    area_of_expertise = profile.get('areaOfExpertise', 'Not specified')

    # Location
    location = profile.get('currentLocation', {})
    location_str = ", ".join(filter(None, [
        location.get('city', ''),
        location.get('state', ''),
        location.get('country', '')
    ])) or "No location specified"

    # Summary
    summary = profile.get('carrierSummary', 'No summary provided.')

    # Education
    education_entries = profile.get('education', [])
    education_str = "\n".join([
        f"- {edu.get('degree', '')} at {edu.get('institute', '')} ({edu.get('startDate', 'N/A')} - {edu.get('endDate', 'N/A')})"
        for edu in education_entries
    ]) or "No education data"

    # Experience
    experience_entries = profile.get('experience', [])
    experience_str = "\n".join([
        f"- {exp.get('position', '')} at {exp.get('company', '')} ({exp.get('startDate', 'N/A')} - {exp.get('endDate', 'N/A')})"
        for exp in experience_entries
    ]) or "No experience data"

    # Skills
    skills = profile.get('highlightedSkills', [])
    skill_str = ", ".join([s.get('name', '') for s in skills]) or "No skills listed"

    # Certifications
    certifications = profile.get('certifications', [])
    cert_str = "\n".join([
        f"- {cert.get('title', '')} from {cert.get('issuer', '')}"
        for cert in certifications
    ]) or "No certifications listed"

    # Role Type and Experience
    role_type = profile.get('roleType', 'Not specified')
    total_experience = profile.get('totalExperience', 'Not specified')

    return f"""
Name: {full_name}
Expertise: {area_of_expertise}
Role Type: {role_type}
Total Experience: {total_experience}
Location: {location_str}
Summary: {summary}

Education:
{education_str}

Experience:
{experience_str}

Highlighted Skills:
{skill_str}

""".strip()

# Main loop
if __name__ == '__main__':
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    try:
        model = init_chat_model("mistral-large-latest", model_provider="mistralai")
        print("🤖 Chat model initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize chat model: {e}")
        exit(1)

    chat_history = ChatMessageHistory()
    print("📌 Chat memory initialized.\n")

    print("🗨️ Ask questions about profiles (e.g., 'Tell me about Arshad Ahmad') or type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("👋 Exiting. Bye!")
            break

        chat_history.add_message(HumanMessage(content=user_input))

        profile_context = search_profile(user_input)

        if profile_context:
            messages = [
                SystemMessage(content="You are a helpful assistant that answers based on the provided profiles."),
                HumanMessage(content=f"{profile_context}\n\nQuestion: {user_input}")
            ]
        else:
            messages = [
                SystemMessage(content="You are a helpful assistant. No profile data found. Say 'No profile found' if needed and try to assist anyway."),
                HumanMessage(content=user_input)
            ]

        print("Assistant: ", end="")
        response_content = ""
        try:
            for token in model.stream(messages):
                print(token.content, end="")
                response_content += token.content
            print()
        except Exception as e:
            print(f"\n❌ Error during response generation: {e}")
            continue

        chat_history.add_message(AIMessage(content=response_content))









