import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")


from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage


prompt = ChatPromptTemplate([("system","you are one piece anime expect "),MessagesPlaceholder(variable_name="history"),("human","{user_input}"),] )



chat_history =[]
chain = prompt | model
while( True):
  user_input = input("user: ")
  if user_input == "bye":
    break
  response =chain.invoke({"history":chat_history, "user_input":user_input})
  chat_history.append(HumanMessage(content=user_input))
  chat_history.append(AIMessage(content=response.content))
  print(f"AI : {response.content}")
print(chat_history)