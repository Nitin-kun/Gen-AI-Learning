from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

loader = TextLoader("./kendrick.txt", encoding="utf-8")

doc = loader.load()

prompt = PromptTemplate(template="explain this in 5 line {doc}",input_variables=["doc"])


chain = prompt | model | parser

response = chain.invoke({"doc":doc})

print(len(doc))

# print(response)
