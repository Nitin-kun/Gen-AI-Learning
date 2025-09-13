from langchain_community.document_loaders import PyPDFLoader
import pypdf 

loader = PyPDFLoader("./kendrick.pdf")

doc = loader.load()

print(len(doc))