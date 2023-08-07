from modules.model import get_llm

import os
from datetime import datetime
import pandas as pd

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from components.postgre_wrapper import PgService
from components.frontend_cost import get_cost_dict

import streamlit as st

ss = st.session_state

def init():
    os.environ["OPENAI_API_KEY"] = st.session_state.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = 'https://mobile-beacon.openai.azure.com/'
    os.environ['OPENAI_API_VERSION'] = '2023-05-15'
    os.environ['OPENAI_API_TYPE'] = 'azure'
    global engine
    global temperature
    global db
    global chat_history
    engine = st.session_state.get("OPENAI_MODEL")
    temperature = st.session_state.get("OPENAI_TEMPERATURE")
    db = ss.get('milvus_instance')
    chat_history = ss.get('chat_with_slack_buffer') if ss.get('chat_with_slack_buffer') else ConversationBufferMemory()

pg_service = PgService()


@pg_service.insert_decorator
def chat_with_slack(query):
    
    init()

    docs = db.similarity_search_with_score(query)

    # control the input to the QA model
    modified_docs = docs.copy()
    modified_docs = [
        Document(
            page_content=f'{x[0].metadata["user"]} form {x[0].metadata["channel"]} on {datetime.fromtimestamp(float(x[0].metadata["timestamp"])).strftime("%B %d, %Y")} at {datetime.fromtimestamp(float(x[0].metadata["timestamp"])).strftime("%H:%M:%S")} said: "{x[0].page_content}"',
            metadata=x[0].metadata
        ) for x in modified_docs
    ]


    llm = get_llm(engine, temperature)

    template = """Given the following a question from human and several extracted parts of a long document.
    Your response should be based on the history of the conversation and the extracted parts of the document.
    IGNORE the source of the extracted parts of the document. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    QUESTION: {question}
    =========
    {summaries}
    =========
    HISTORY: {chat_history}
    =========
    ANSWER:"""

    PROMPT = PromptTemplate(template=template, input_variables=['question', 'summaries', 'chat_history'])

    qa_chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type='stuff',
        prompt=PROMPT,
        # memory = chat_history,
        verbose=True,
    )

    with get_openai_callback() as cb:
        response = qa_chain({'input_documents': modified_docs, 'question': query, 'chat_history': chat_history.buffer})
        print(cb)
    
    filtered_response = response['output_text'].replace('<', '').replace('>', '')

    chat_history.save_context({'input': query}, {'output': filtered_response})

    ss['chat_with_slack_buffer'] = chat_history

    md_template = """{count}
    Similarity distance : {score}
    Content : {content}
    User : {user}
    Channel : {channel}
    Timestamp : {timestamp}
    Chunk : {chunk}"""

    sources = [{
        'score': round(doc[1], 5),
        'content': doc[0].page_content,
        'user': doc[0].metadata["user"],
        'channel': doc[0].metadata["channel"],
        'timestamp': datetime.fromtimestamp(float(doc[0].metadata["timestamp"])).strftime("%Y/%m/%d, %H:%M:%S"),
        'chunk': doc[0].metadata["chunk"]                      
    } for doc in docs]

    ss['chat_with_slack_sources'] = [sources] + [[]] + ss.get('chat_with_slack_sources', [])
    ss['chat_with_slack_cost'] = [get_cost_dict(cb)] + [{}] + ss.get('chat_with_slack_cost', [])
    return cb