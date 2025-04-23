import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

class AIAGENT:
    def __init__(self):
        self.model = self.initialize_model()
        self.tools= self.initialize_tools()

    def initialize_tools(self):
        tavily_search = TavilySearchResults(max_results=2)
        print("Tools initialized.")
        return [tavily_search]

    def initialize_model(self):
        model = init_chat_model("mistral-large-latest", model_provider="mistralai")
        print("Model Initialized successfully !! ")
        return model

    def check_keys(self):
        langsmith= os.getenv("LANGSMITH_API_KEY")
        tavily= os.getenv("TAVILY_API_KEY")
        mistral =  os.getenv("MISTRAL_API_KEY")

        if langsmith and tavily and mistral:
            print("Keys are loaded successfully")
        else :
            print("Keys are not loaded in the env...")

    def basic_invoke(self, user_message ="hii"):
        response = self.model.invoke([HumanMessage(content=user_message)])
        print(f"Response: {response.content}")

    # def model_initialization(self):
    #     response = self.model.invoke([HumanMessage(content="hi!")])
    #     response.content
    #     print(response.content,"R..**********")

    def model_initialization_with_tools(self):
        ## .bind_tools to give the language model knowledge of these tools
        model_with_tools = self.model.bind_tools(self.tools)
        response = model_with_tools.invoke([HumanMessage(content="What is the highest percentage of finals of JEE?!")])
        print(f"ContentString: {response.content}")
        print(f"ToolCalls: {response.tool_calls}")

    def create_the_agent(self):
        agent_executor = create_react_agent(self.model, self.tools)
        # response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content="whats the weather in dubai?")]}
        )
        print(response["messages"])


    def agent_with_streaming_response(self):
        agent_executor = create_react_agent(self.model, self.tools)

        for step in agent_executor.stream(
                    {"messages": [HumanMessage(content="whats the population of today's india?")]},
                    stream_mode="values",
            ):
                step["messages"][-1].pretty_print()

    def ai_Agent_with_memory(self):
        memory = MemorySaver()
        agent_executor = create_react_agent(self.model, self.tools, checkpointer=memory)
        config = {"configurable": {"thread_id": "abc123"}}

        for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content="Hello, I'm alice Keep in mind i willa sk you about my name?")]}, config
        ):
            print(chunk)
            print("----")

        for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content="whats my name?")]}, config
        ):
            print(chunk)
            print("----")




if __name__=="__main__":
    ai = AIAGENT()
    ai.check_keys()
    # model_initialization()
    # model_initialization_with_tools()
    # create_the_agent()
    # ai.agent_with_streaming_response()
    ai.ai_Agent_with_memory()


