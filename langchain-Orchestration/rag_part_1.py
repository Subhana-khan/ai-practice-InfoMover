# import getpass
# import os
# from langchain_core.vectorstores import InMemoryVectorStore
# import bs4
# from langchain import hub
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict
# from langchain.chat_models import init_chat_model
# from langchain_mistralai import MistralAIEmbeddings
#
# # --- 1. Configuration ---
# def configure_api_key():
#     if not os.environ.get("MISTRAL_API_KEY"):
#         os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")
#
#
# # --- 2. Document Loader & Chunking ---
# def load_and_split_documents(url: str) -> List[Document]:
#     loader = WebBaseLoader(
#         web_paths=(url,),
#         bs_kwargs=dict(
#             parse_only=bs4.SoupStrainer(
#                 class_=("post-content", "post-title", "post-header")
#             )
#         ),
#     )
#     documents = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_documents(documents)
#
#
# # --- 3. Vector Store Setup ---
# def setup_vectorstore(docs: List[Document]) -> InMemoryVectorStore:
#     embeddings = MistralAIEmbeddings(model="mistral-embed")
#     store = InMemoryVectorStore(embeddings)
#     store.add_documents(docs)
#     return store
#
# # Define state for application
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str
#
# # Define application steps
# # def retrieve(state: State):
# #     retrieved_docs = setup_vectorstore.similarity_search(state["question"])
# #     return {"context": retrieved_docs}
#
# def retrieve(state: State, store: InMemoryVectorStore) -> dict:
#     results = store.similarity_search(state["question"])
#     return {"context": results}
#
# def generate(state: State):
#     llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}
#
#
# # Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()
# response = graph.invoke({"question": "What is Task Decomposition?"})
# print(response["answer"])
#
# if __name__== "__main__":
#     configure_api_key()
#     prompt = hub.pull("rlm/rag-prompt")
#     url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
#     docs = load_and_split_documents(url)
#     vector_store = setup_vectorstore(docs)
#





import getpass
import os
import bs4
from typing_extensions import List, TypedDict

# LangChain imports
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain import hub
from langgraph.graph import START, StateGraph


# STEP 1: Ask for API key if not already set
def set_api_key():
    if "MISTRAL_API_KEY" not in os.environ:
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter Mistral AI API key: ")


# STEP 2: Load content from URL and split it into smaller chunks
def load_blog_and_split(url: str) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    return split_docs


#  STEP 3: Store chunks into memory using embeddings
def create_memory_store(split_docs: List[Document]) -> InMemoryVectorStore:
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(split_docs)
    return vector_store


#  State format used in the app
# For a simple RAG application, we can just keep track of the input question,
# retrieved context, and generated answer:
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


#  STEP 4a: Retrieve relevant docs from memory using question

# Our retrieval step simply runs a similarity search using the input question,
# and the generation step formats the retrieved context and original question into a prompt
# for the chat model.

# Nodes (application steps) (a)retrieve_step (b)generate_step

def retrieve_step(state: State, store: InMemoryVectorStore) -> dict:
    matching_docs = store.similarity_search(state["question"])
    return {"context": matching_docs}


# STEP 4b: Generate an answer using the context and question
def generate_step(state: State, prompt_template, llm_model) -> dict:
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt_template.invoke({"question": state["question"], "context": context_text})
    response = llm_model.invoke(messages)
    return {"answer": response.content}


#  STEP 5: Main function to execute everything
def main():
    set_api_key()

    # Load the prompt template and question-answering model
    prompt_template = hub.pull("rlm/rag-prompt")
    llm_model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    # Step-by-step flow
    blog_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    split_documents = load_blog_and_split(blog_url)
    memory_store = create_memory_store(split_documents)

    # Finally, we compile our application into a single graph object. In this case, we are just
    # connecting the retrieval and generation steps into a single sequence.

    # Create the LangGraph workflow
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", lambda state: retrieve_step(state, memory_store))
    graph_builder.add_node("generate", lambda state: generate_step(state, prompt_template, llm_model))
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()

    # Ask a question
    question = "What is Task Decomposition?"
    result = graph.invoke({"question": question})
    print("\nAnswer:\n", result["answer"])


# ▶ Run the app
if __name__ == "__main__":
    main()
