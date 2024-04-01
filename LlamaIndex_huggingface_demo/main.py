# https://docs.llamaindex.ai/en/latest/getting_started/starter_example_local/

# pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-m3")

# ollama
Settings.llm = Ollama(model="gemma", request_timeout=60.0)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


# query engine
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
# response = query_engine.query("What did the author do growing up?")
print(response)
