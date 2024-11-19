import os
from langchain.embeddings import VertexAIEmbeddings

credential_path = ""

embeddings = VertexAIEmbeddings()

text = "Where is Tai Wong Ha Resite Village RCP and there is a recycling point?"

query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])

print(doc_result)