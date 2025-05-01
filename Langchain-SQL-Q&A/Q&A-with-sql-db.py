from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

keys = os.getenv("TOGETHER_API_KEY")
llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 2
for message in query_prompt_template.messages:
    message.pretty_print()

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

# convert question to query
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

write_query_response= write_query({"question": "How many Employees are there?"})
print(write_query_response,"WQ")

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}
execute_query_response = execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})
print(execute_query_response,"EQ")


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()
for step in graph.stream(
        {"question": "How many employees are there?"}, stream_mode="updates"
):
    print(step,"final response")


