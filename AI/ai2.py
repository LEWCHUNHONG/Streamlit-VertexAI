import langchain, os
from langchain.llms import VertexAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents import initialize_agent, Tool,load_tools, AgentType, Agent


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
credential_path = ""

embeddings = VertexAIEmbeddings()
template = """Question: {question}"""
query_result = embeddings.embed_query(template)

doc_result = embeddings.embed_documents([template])
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

urls = [
    "https://www.wastereduction.gov.hk/sites/default/files/wasteless07.csv"
]

loader = UnstructuredURLLoader(urls=urls, limit=5)
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
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


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, limit=5 )
agent.run(prompt="Where is Tai Wong Ha Resite Village RCP and there is a recycling point?")
