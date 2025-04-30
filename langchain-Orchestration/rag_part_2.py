
import os
import bs4
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_mistralai import MistralAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

class RagRetrieverWithTools:
    def __init__(self):
        self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.vector_store = None  # will be set later after loading blog
        self.retrieve = self.create_tool()  # ✅ Add this line


    def load_blog_and_split(self, url: str) -> List[Document]:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
        )
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        return split_docs

    def create_memory_store(self, split_docs: List[Document]):
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(split_docs)


    def create_tool(self):
        #"""" @tool: Converts a function into a callable "tool" the LLM can invoke."""
        @tool(response_format="content_and_artifact")
        def retrieve( query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        return retrieve

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate final answer using tool message content."""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
               or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def build_graph(self):
        memory = MemorySaver()

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", ToolNode([self.retrieve]))
        graph_builder.add_node("generate", self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        graph = graph_builder.compile(checkpointer=memory)
        return graph


if __name__ == "__main__":
    rag = RagRetrieverWithTools()

    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    split_docs = rag.load_blog_and_split(url)
    rag.create_memory_store(split_docs)

    graph = rag.build_graph()

    input_message = "hii, I'm bca degree student"
    config = {"configurable": {"thread_id": "abc123"}}

    for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            config = config,
            stream_mode="values",
    ):
        print(step["messages"][-1].content)



# Documentation
# MessagesState: Maintains message history for LangGraph.
#
# StateGraph: The heart of LangGraph’s logic flow.
#
# ToolNode: Executes retrieval tool when called.
#
# tools_condition: Tells the graph when to use tools or not.