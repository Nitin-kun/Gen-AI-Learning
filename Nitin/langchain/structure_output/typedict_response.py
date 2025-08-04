import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from typing import TypedDict

load_dotenv()

class review(TypedDict):
    '''Summarize the review and its sentiment '''
    summery:str
    sentiment:str


model = ChatMistralAI(model="mistral-large-latest")
model_with_structure = model.with_structured_output(review)


response = model_with_structure.invoke("the product was good and the screen was perfecta nd bright and vibrant and speaker ws lit")
print("--"*95)


print(response)

