# pip install llama-index
# pip install python-dotenv

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load the environment variables
load_dotenv()

# Now you can access the variables using os.getenv
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)