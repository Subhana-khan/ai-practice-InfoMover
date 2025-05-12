# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain.chains.summarize import load_summarize_chain
# from langchain.docstore.document import Document
# from langchain.text_splitter import CharacterTextSplitter
# import os
# import getpass
#
# # Step 1: Prompt for Mistral API Key (before using ChatMistralAI)
# def set_api_key():
#     if "MISTRAL_API_KEY" not in os.environ:
#         os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter Mistral AI API key: ")
#
# set_api_key()
#
# # Step 2: Read the long text
# with open("text-file.txt") as f:
#     text_file = f.read()
#
# # Step 3: Split text into smaller chunks
# text_splitter = CharacterTextSplitter()
# texts = text_splitter.split_text(text_file)
#
# # Step 4: Convert each chunk into Document object
# docs = [Document(page_content=t) for t in texts]
#
# # Step 5: Initialize the Mistral model
# llm = ChatMistralAI(model="mistral-small", temperature=0)
#
# # Step 6: Load summarization chain using map-reduce
# chain = load_summarize_chain(llm, chain_type="map_reduce")
#
# # Step 7: Use `.invoke()` instead of deprecated `.run()`
# summary = chain.invoke({"input_documents": docs})
#
# print("\n=== Final Summary ===\n")
# print(summary["output_text"])



# from langchain_community.document_loaders import WebBaseLoader
# from langchain.chat_models import init_chat_model
# from langchain_core.prompts import ChatPromptTemplate
# import operator
# from typing import Annotated, List, Literal, TypedDict
# from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
# from langchain_core.documents import Document
# from langgraph.constants import Send
# from langgraph.graph import END, START, StateGraph
# import os
# from dotenv import load_dotenv
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Retrieve the OpenAI API key from the environment variables
# openai_api_key = os.getenv("GROQ_API_KEY")
#
# if not openai_api_key:
#     raise ValueError("GROQ_API_KEY environment variable not set.")
#
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# docs = loader.load()
#
# # llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# llm = init_chat_model("llama3-8b-8192", model_provider="groq")
#
#
#
# map_prompt = ChatPromptTemplate.from_messages(
#     [("system", "Write a concise summary of the following:\n\n{context}")]
# )
#
# reduce_template = """
# The following is a set of summaries:
# {docs}
# Take these and distill it into a final, consolidated summary
# of the main themes.
# """
# reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
#
#
# token_max = 1000
#
# def length_function(documents: List[Document]) -> int:
#     """Get number of tokens for input contents."""
#     return sum(llm.get_num_tokens(doc.page_content) for doc in documents)
#
# class OverallState(TypedDict):
#     contents: List[str]
#     summaries: Annotated[list, operator.add]
#     collapsed_summaries: List[Document]
#     final_summary: str
#
# class SummaryState(TypedDict):
#     content: str
#
# async def generate_summary(state: SummaryState):
#     prompt = map_prompt.invoke(state["content"])
#     response = await llm.ainvoke(prompt)
#     return {"summaries": [response.content]}
#
# def map_summaries(state: OverallState):
#     return [
#         Send("generate_summary", {"content": content}) for content in state["contents"]
#     ]
#
# def collect_summaries(state: OverallState):
#     return {
#         "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
#     }
#
# async def _reduce(input: dict) -> str:
#     prompt = reduce_prompt.invoke(input)
#     response = await llm.ainvoke(prompt)
#     return response.content
#
# async def collapse_summaries(state: OverallState):
#     doc_lists = split_list_of_docs(
#         state["collapsed_summaries"], length_function, token_max
#     )
#     results = []
#     for doc_list in doc_lists:
#         results.append(await acollapse_docs(doc_list, _reduce))
#
#     return {"collapsed_summaries": results}
#
# def should_collapse(
#         state: OverallState,
# ) -> Literal["collapse_summaries", "generate_final_summary"]:
#     num_tokens = length_function(state["collapsed_summaries"])
#     if num_tokens > token_max:
#         return "collapse_summaries"
#     else:
#         return "generate_final_summary"
#
# async def generate_final_summary(state: OverallState):
#     response = await _reduce(state["collapsed_summaries"])
#     return {"final_summary": response}
#
# graph = StateGraph(OverallState)
# graph.add_node("generate_summary", generate_summary)
# graph.add_node("collect_summaries", collect_summaries)
# graph.add_node("collapse_summaries", collapse_summaries)
# graph.add_node("generate_final_summary", generate_final_summary)
#
# graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
# graph.add_edge("generate_summary", "collect_summaries")
# graph.add_conditional_edges("collect_summaries", should_collapse)
# graph.add_conditional_edges("collapse_summaries", should_collapse)
# graph.add_edge("generate_final_summary", END)
#
# app = graph.compile()
#
# from langchain_text_splitters import CharacterTextSplitter
#
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=1000, chunk_overlap=0
# )
# split_docs = text_splitter.split_documents(docs)
# print(f"Generated {len(split_docs)} documents.")
#
# result = app.invoke({"contents": [doc.page_content for doc in split_docs]})
# print(result["final_summary"])




import os
import asyncio
from dotenv import load_dotenv
import operator
from typing import Annotated, List, Literal, TypedDict
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize the language model
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
# llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")
# llm = init_chat_model("mistral-large-latest", model_provider="mistralai")


# Define prompts
# Map step: Summarize each chunk.
map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\n\n{context}")]
)

# Reduce step: Merge multiple summaries into one.
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

# Load documents
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Split documents
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
print(f"Generated {len(split_docs)} documents.")

token_max = 1000

def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str

async def generate_summary(state: SummaryState):
    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}

def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]

def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }

# """this func is used to merge all small docs or summaries into one""""
async def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content

async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}

def should_collapse(
        state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

async def generate_final_summary(state: OverallState):
    response = await _reduce(state["collapsed_summaries"])
    return {"final_summary": response}

graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()

# Run the application
async def main():
    result = await app.ainvoke({"contents": [doc.page_content for doc in split_docs]})
    print(result["final_summary"])

asyncio.run(main())

# flow of the above code

# START
# ↓
# map_summaries
# ↓
# generate_summary
# ↓
# collect_summaries
# ↓
# ┌───────────── collapse_summaries (if too big)
# ↓                              ↓
# generate_final_summary ←────────┘
# ↓
# END