from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

parser = StrOutputParser()

passthrough = RunnablePassthrough()


model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

joke_prompt = PromptTemplate(
    template="create a funny joke on {topic}",
    input_variables=["topic"]
)

explain_prompt = PromptTemplate(
    template="explain this joke to a 5 year old {joke}",
    input_variables=["joke"]
)



joke_generator = RunnableSequence(joke_prompt , model , parser)

joke_explainer  = RunnableParallel({"joke":passthrough , "explanation": RunnableSequence(explain_prompt,model,parser)})

final_chain = RunnableSequence(joke_generator,joke_explainer)


print(final_chain.invoke({"topic":"cat"}))
