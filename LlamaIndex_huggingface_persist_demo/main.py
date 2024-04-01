# https://docs.llamaindex.ai/en/latest/getting_started/starter_example_local/

# pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-m3")

# ollama
Settings.llm = Ollama(model="gemma", request_timeout=60.0)

index = VectorStoreIndex.from_documents(
    documents,
)

# query engine
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
# response = query_engine.query("Was hat der Autor in seiner Jugend gemacht?")
print(response)