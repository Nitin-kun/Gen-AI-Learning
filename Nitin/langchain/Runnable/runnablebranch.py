from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="My BMI is {bmi}. Give me motivation to be fit.",
    input_variables=["bmi"]
)

chain = RunnableBranch(
    (lambda x: float(x["bmi"]) > 24.5, RunnableSequence(prompt1 | model | parser)),
    (lambda x: float(x["bmi"]) < 18.5, RunnableSequence(prompt1 | model | parser)),
    RunnableSequence(
        PromptTemplate(template="You are fit!", input_variables=[]) | model | parser
    )
)

response = chain.invoke({"bmi": 19})
print(response)
