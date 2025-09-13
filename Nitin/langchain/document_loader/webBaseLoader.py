from langchain_community.document_loaders import WebBaseLoader


url = 'https://medium.com/@vikrampande783/introduction-to-langchain-9e09aae37e62'

loader = WebBaseLoader(url)

print(loader.load())
