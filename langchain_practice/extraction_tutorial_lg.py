
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

def load_key():
    key = os.getenv("MISTRAL_API_KEY")
    if key:
        print("Key is successfully loaded from environment variables.")
    else:
        raise ValueError("Key not found. Please set it in the .env file.")


class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The hair_color of the person if known"
    )
    height: Optional[str] = Field(
        default=None, description="Convert the height in meters"
    )

def prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{input}"),
        ]
    )

def run_classification(input_text: str):
    llm = ChatMistralAI(temperature=0, model="mistral-large-latest")
    prompt = prompt_template().invoke({"input": input_text})
    structured_llm = llm.with_structured_output(schema=Person)

    response = structured_llm.invoke(prompt)
    print("Input:", input_text)
    print("Output:", response)

if __name__ == "__main__":
    load_key()
    input_text = "His name is John & has 6 feet height with beautiful brown hair"
    run_classification(input_text)

# output: Output: name='John' hair_color='brown' height='6 feet'