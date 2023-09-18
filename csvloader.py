
from langchain.document_loaders import UnstructuredURLLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool,load_tools, AgentType, Agent




urls = [
    "https://www.wastereduction.gov.hk/sites/default/files/wasteless07.csv"
]

loader = UnstructuredURLLoader(urls=urls, limit=10)
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)
docsearch = Chroma.from_documents(texts, embeddings, collection_name="bq-wasteless")

bqtech = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

tools = [
    Tool(
        name="bq-wasteless",
        func=bqtech.run,
        description="useful for when you need to answer  technical questions about BigQuery .",
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Where is Tai Wong Ha Resite Village RCP and there is a recycling point?")
