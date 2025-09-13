from dotenv import load_dotenv
import os 
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

parser = StrOutputParser()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model  = GoogleGenerativeAI(model = "gemini-2.5-flash")

prompt1  = PromptTemplate(template="give me the protein in 100g of {item}" , input_variables=['item'])

runner = RunnableSequence(prompt1, model, parser)

response = runner.invoke({'item':"chicken breast"})

print(response)

graph = runner.get_graph()
graph.print_ascii()
