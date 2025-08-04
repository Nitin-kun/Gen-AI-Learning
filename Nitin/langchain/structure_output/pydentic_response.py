import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

load_dotenv()




class review(BaseModel):
    summery:str 
    sentiment:str

model = ChatMistralAI(model= "mistral-large-latest")
model_with_structure = model.with_structured_output(review)
response = model_with_structure.invoke("the product was good and the screen is perfect with great sound quality ")

print()
print(response)
