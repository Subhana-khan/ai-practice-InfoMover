
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

load_dotenv()

def load_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        print("OpenAI API key loaded successfully.")
    else:
        raise ValueError("OpenAI API key not found in .env file.")

# === Prompt Template (Common for both cases) ===
def get_prompt():
    return ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage.
        Only extract the properties mentioned in the 'Classification' function.

        Passage:
        {input}
        """
    )

# ----------- 1. Simple Classification Schema -----------
class SimpleClassification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(description="Aggression level from 1 to 10")
    language: str = Field(description="Language of the text")

# -------------- 2. Finer-Control Classification Schema ---------------
class FinerClassification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="How aggressive the statement is (1-5)",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

# ------------------ Function to run classification--------------------
def run_classification(input_text: str, mode: str = "finer"):
    if mode == "simple":
        model_schema = SimpleClassification
    else:
        model_schema = FinerClassification

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(model_schema)
    prompt = get_prompt().invoke({"input": input_text})
    response = llm.invoke(prompt)

    print(f"\n Classification Mode: {mode.upper()}")
    print("Input:", input_text)
    print("Output:", response.model_dump())

if __name__ == "__main__":
    load_api_key()

    # Change text and mode here to switch between classification types
    input_text = "Vamos al cine esta noche"
    # run_classification(test_input, mode="simple")
    run_classification(input_text, mode="finer")
