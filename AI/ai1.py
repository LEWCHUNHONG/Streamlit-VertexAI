import langchain, os
from langchain.llms import VertexAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
credential_path = ""

template = """Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Who is Messi?"
print(llm_chain(question))