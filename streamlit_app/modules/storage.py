from pymilvus import connections, utility
import time
import streamlit as st
import os

ss = st.session_state

from modules.embedding import embed_with_azure_openai
from modules.model import get_embedding

from modules.slack_reader import SlackDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Milvus

MILVUS_HOST='52.226.226.29'
milvus_connection_args = {'host': MILVUS_HOST}

def init():
    os.environ["OPENAI_API_KEY"] = st.session_state.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = 'https://mobile-beacon.openai.azure.com/'
    os.environ['OPENAI_API_VERSION'] = '2023-05-15'
    os.environ['OPENAI_API_TYPE'] = 'azure'
    global engine
    global temperature
    global embedding_egine
    engine = st.session_state.get("OPENAI_MODEL")
    embedding_egine = st.session_state.get("OPENAI_EMBEDDING_MODEL")
    temperature = st.session_state.get("OPENAI_TEMPERATURE")

def get_all_collections():
    connections.connect(host=MILVUS_HOST, port="19530")
    collections = utility.list_collections()
    collections = [x for x in collections if x.startswith('dev_slack_')]
    collections.sort()
    # connections.disconnect('default')
    return collections

def get_slack_docs(zip_file, file_name):
    loader = SlackDirectoryLoader(zip_file, file_name)
    docs = loader.load()
    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    doc_chunks = []

    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            metadata_copy = doc.metadata.copy()
            metadata_copy['chunk'] = i
            doc = Document(
                page_content=chunk, metadata=metadata_copy
            )
            doc_chunks.append(doc)
            
    return doc_chunks

def store_to_milvus_vector_database(docs, model, collection_name='temp'):
    timestamp = int(time.time() * 1000.0)
    collection_name = f"dev_slack_{collection_name}_{timestamp}"
    doc_chunks = split_docs(docs)
    embeddings = get_embedding(model)

    index = []
    for doc_chunk in doc_chunks:
        index.append(embed_with_azure_openai([doc_chunk], embeddings, collection_name))

    return index

def get_milvus_instance():

    init()

    collection_name = ss.get('selected_collection')

    embeddings = get_embedding(embedding_egine)
    db = Milvus(
        collection_name=collection_name,
        embedding_function=embeddings,
        connection_args=milvus_connection_args,
    )
    ss['milvus_instance'] = db
