import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

# Load web content
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
print(loader,"loader")

docs = loader.load()
print("****")

# llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)
print(prompt,"prompt here")

# Instantiate chain
chain = create_stuff_documents_chain(llm, prompt)

# Invoke chain
result = chain.invoke({"context": docs})
print(result)