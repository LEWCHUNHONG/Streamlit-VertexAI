import vertexai, os
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextGenerationModel
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspaces/project/fyp-open-data-hackathon-7fccdf48c91c.json"
credential_path = "/workspaces/project/fyp-open-data-hackathon-7fccdf48c91c.json"
vertexai.init(project="fyp-open-data-hackathon", location="us-central1")


generation_model = TextGenerationModel.from_pretrained("text-bison@001")

agent = create_csv_agent(
    generation_model(temperature=0, model="text-bison@001"),
    "https://www.immd.gov.hk/opendata/hkt/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run("how many rows are there?")



#generation_model = TextGenerationModel.from_pretrained("text-bison@001")
#response = generation_model.predict(
 ##  max_output_tokens= 256,
   # temperature= 0,
    #top_p= 0.8,
    #top_k= 40
#)
#print(response.text)
