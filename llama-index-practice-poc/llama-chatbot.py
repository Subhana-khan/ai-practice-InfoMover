from dotenv import load_dotenv
import os
from llama_index.llms.together import TogetherLLM
#
from langchain_openai import OpenAI
from llama_index.core import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    PromptHelper,
    StorageContext,
    load_index_from_storage,
    Settings
)
load_dotenv()

# Set OpenAI API key
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

# Set working directory
os.chdir(r"C:\Users\Lenovo\OneDrive\Desktop\ai-practice\llama-index-practice-poc")

# Paths
DOCUMENTS_DIR = "data"
INDEX_DIR = "index_storage"

def load_documents(directory_path):
    return SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".txt", ".pdf", ".md"]
    ).load_data()

def configure_settings():
    # Set LLM and prompt helper in global Settings

    # Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=512)

    Settings.llm = TogetherLLM(
            model="mistral-7b-instruct",
            api_key=os.getenv("TOGETHER_API_KEY")
        )
    Settings.prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1
    )

def create_or_load_index(documents):
    if os.path.exists(INDEX_DIR):
        print("[INFO] Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("[INFO] Creating new index and saving to disk...")
        index = GPTVectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        return index

def query_index(index, query):
    response = index.query(query)
    return response.response

def main():
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"[ERROR] Directory '{DOCUMENTS_DIR}' not found.")
        return

    configure_settings()
    documents = load_documents(DOCUMENTS_DIR)
    index = create_or_load_index(documents)

    print("\n Chatbot is ready! Ask questions about your documents.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        answer = query_index(index, user_input)
        print("Chatbot:", answer)

if __name__ == "__main__":
    main()
