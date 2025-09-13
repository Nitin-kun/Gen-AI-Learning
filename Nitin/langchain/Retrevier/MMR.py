from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
import os 
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="The Eiffel Tower is located in Paris, France."),
    Document(page_content="The Great Wall of China is visible from space only with aid."),
    Document(page_content="Mount Everest is the highest mountain above sea level."),
    Document(page_content="Python is a popular programming language for AI and data science."),
    Document(page_content="LangChain provides tools to build applications powered by LLMs.")
]

print(os.getenv("GOOGLE_API_KEY"))
 
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(docs, embedding)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# Depending on your LangChain version:
result = retriever.invoke("tell me something about famous landmarks")


for i, doc in enumerate(result, 1):
    print(f"Result {i}:\n{doc.page_content}\n")
