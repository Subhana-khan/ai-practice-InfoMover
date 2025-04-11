from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

from dotenv import load_dotenv


load_dotenv()
# ------------------------------------Loading documents----------------------------------------------------

file_path = "https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# print(len(docs))
# output : 106

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# output:FORM 10-KFORM 10-K
# {'producer': 'Wdesk Fidelity Content Translations Version 008.001.016', 'creator': 'Workiva', 'creationdate':
# '2023-07-20T22:09:22+00:00', 'author': 'anonymous', 'moddate': '2023-0
#  7-26T15:13:52+08:00', 'title': 'Nike 2023 Proxy', 'source': 'https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf',
#  'total_pages': 106, 'page': 0, 'page_label': '1'}


# --------------------------------------------Splitting----------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits,"all splits count"))
# output : 501

# print(all_splits[0], "1st split")

# output : # page_content='FORM 10-KFORM 10-K' metadata={'producer': 'Wdesk Fidelity Content Translations Version 008.001.016',
# 'creator': 'Workiva', 'creationdate': '2023-07-20T22:09:22+00:00'
# ,'author': 'anonymous', 'moddate': '2023-07-26T15:13:52+08:00', 'title': 'Nike 2023 Proxy',
# 'source': 'https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf',
# 'total_pages': 106, 'page': 0, 'page_label': '1', 'start_index': 0} 1st split


# --------------------------------------------Embeddings-----------------------------------------------------

# Retrieve the Hugging Face token from environment variables
# hf_token = os.getenv("HUGGING_FACE_API_KEY")
ms_token =os.getenv("MISTRAL_API_KEY")
if ms_token:
    print("MS token successfully loaded from environment variables.")
else:
    raise ValueError("MS token not found. Please set it in the .env file.")

# Initialize the OpenAIEmbeddings with the model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=ms_token)


# Assuming all_splits is defined somewhere in your code
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

# Check if the vectors have the same length
assert len(vector_1) == len(vector_2)

# print(f"Generated vectors of length {len(vector_1)}\n")

# output : Generated vectors of length 1024

# print(vector_1[:10],"vector_1")

# output : [-0.031341552734375, 0.0247802734375, 0.017730712890625, -0.00920867919921875, 0.033905029296875,
#           0.043304443359375, 0.0216522216796875, 0.01126861572265625, -0.012542724609375, -0.0093536376953125]
#           vector_1

# -----------------------------------------Vector stores------------------------------------------------------

vector_store = InMemoryVectorStore(embeddings)
# print(vector_store, "vector_store")

# output : <langchain_core.vectorstores.in_memory.InMemoryVectorStore object at 0x000002795F13D3C0> vector_store

ids = vector_store.add_documents(documents=all_splits)
# print(ids, "ids")

# output : 'c2c6a96c-af77-43bd-982b-7bcc9ade1f1e', '9b826856-e406-427b-950b-732677e90b91',
# '1ac7f0c2-941a-41c8-b729-e695b57393d3'................many more, ids]


# ----------------------------------query the db--------------------------------------------------------

# "string query"

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

# print(results[0])

# output : U.S. RETAIL STORES NUMBER
# NIKE Brand factory stores  213
# NIKE Brand in-line stores (including employee-only stores)  74
# Converse stores (including factory stores)  82
# TOTAL  369
# In the United States, NIKE has eight significant distribution centers. Refer to Item 2. Properties for further information.
#     NIKE, INC.       2' metadata={'producer': 'Wdesk Fidelity Content Translations Version 008.001.016', 'creator': 'Workiva', 'creationdate': '2023-07-20T22:09:22+00:00', 'author': 'a
# nonymous', 'moddate': '2023-07-26T15:13:52+08:00', 'title': 'Nike 2023 Proxy', 'source': 'https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf', 'total_pages': 106, 'page': 5, 'page_label': '6', 'start_index': 3218}



# -------------------------------------retireval--------------------------------------------------



@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
