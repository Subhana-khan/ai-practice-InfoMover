from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

# Initialize database and LLM
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# Define State structure
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# Step 1: Generate SQL query from question
def write_query(state: State):
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"],
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)

    # print(" Prompt given to LLM:\n", prompt)
    # print(" LLM Generated SQL:\n", result)

    return {"query": result["query"]}
# Step 2: Execute query
def execute_query(state: State):
    tool = QuerySQLDatabaseTool(db=db)
    # print(tool, "tool")
    return {"result": tool.invoke(state["query"])}

# Step 3: Generate human-readable answer
def generate_answer(state: State):
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    # print(prompt,"prompt here")
    response = llm.invoke(prompt)
    final_response =  {"answer": response.content}
    # print(final_response,"****")
    return final_response

    # print(response,"response")
    # return {"answer": response.content}

# Build LangGraph sequence
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
# print(graph_builder,"gb")
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()
# print(graph, "graph response")

# Interactive Chat Loop
def run_chatbot():
    print("SQL Chatbot is running. Ask your questions about the database.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input(" You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        inputs = {"question": user_input}
        final_output = "final_response"
        for  step in graph.stream(inputs):
            final_output = step
        print(step, "Final_response_is_here")

if __name__ == "__main__":
    run_chatbot()
