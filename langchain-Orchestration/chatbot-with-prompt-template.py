import os
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from langchain_together import ChatTogether
from dotenv import load_dotenv

class ChatbotWithTemplate:
    def __init__(self):
        self.load_config()
        self.model = self.initialize_model()
        self.prompt_template = self.create_prompt_template()
        self.workflow = self.setup_workflow()
        self.thread_id = "unique-thread-id"
        self.config_dict = {"configurable": {"thread_id": self.thread_id}}

    def load_config(self):
        load_dotenv()
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in .env file.")
        print("TOGETHER_API_KEY loaded successfully.")

    def initialize_model(self):
        return ChatTogether(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=512,
            together_api_key=self.api_key
        )

    def create_prompt_template(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You talk like a lawyer. Answer all questions to the best of your ability",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt

    def setup_workflow(self):
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self.call_model)
        workflow.set_entry_point("model")
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def call_model(self, state: MessagesState):
        formatted_messages = self.prompt_template.format_messages(messages=state["messages"])
        response = self.model.invoke(formatted_messages)
        return {"messages": response}


    def get_response(self, user_input):
        input_messages = [HumanMessage(content=user_input)]
        response = self.workflow.invoke({"messages": input_messages}, self.config_dict)
        return response["messages"][-1].content

    def start_conversation(self):
        print("Chatbot is ready! Type 'quit' to exit.")
        while True:
            user_input = input("YOU: ")
            if user_input.lower() == "quit":
                print("Exiting!")
                break
            response = self.get_response(user_input)
            print("Chatbot:", response)

if __name__ == "__main__":
    chatbot = ChatbotWithTemplate()
    chatbot.start_conversation()
