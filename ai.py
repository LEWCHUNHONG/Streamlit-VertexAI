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
prompt = """
Who is Messi?
"""

response = generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=1,
        top_p= 0.8,
        top_k= 40
    ).text

print(response)
