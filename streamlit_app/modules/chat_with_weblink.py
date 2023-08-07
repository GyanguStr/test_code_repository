from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationBufferMemory
from langchain.vectorstores import Milvus
import pandas as pd
from modules.model import get_llm
import streamlit as st
import os
from components.key_vault import FetchKey
from modules.mass_milvus import MassMilvus
from modules.model import get_embedding
from pymilvus import connections, utility, Collection, exceptions
from components.frontend_cost import get_cost_dict
from langchain.callbacks import get_openai_callback
from components.html_templates import user_template, bot_template, css


ss = st.session_state

def init():
    os.environ["OPENAI_API_KEY"] = st.session_state.get("OPENAI_API_KEY", FetchKey("OPENAI-KEY").retrieve_secret())
    os.environ["OPENAI_API_BASE"] = 'https://mobile-beacon.openai.azure.com/'
    os.environ['OPENAI_API_VERSION'] = '2023-05-15'
    os.environ['OPENAI_API_TYPE'] = 'azure'
    
    global MILVUS_HOST
    global milvus_connection_args
    MILVUS_HOST='52.226.226.29'
    milvus_connection_args = {'host': MILVUS_HOST}


# -----BACKEND-----
@st.cache_data
def load_urls(urls: list = []):
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    return data

@st.cache_data
def split_data(_doc: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
    )
    docs = text_splitter.split_documents(_doc)
    return docs

@st.cache_data
def get_summary(_docs):
    init()

    llm = get_llm('GPT-4', 0)
    
    map_prompt_template = """Write a concise summary of the text below.
    Avoid summarizing code like content.
    The final consise summary should be no longer than 1 phrase.

    TEXT : {text}

    CONCISE SUMMARY:"""

    combine_prompt_template = """Write a concise summary of the following text between the triple backticks.
    The final concise summary should be less than 15 words.
    The final concise summary should be easily understood by a 10-year-old child.
    Don't include any triple backticks in the final concise summary.

    ```{text}```

    CONCISE SUMMARY:"""

    MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce", 
        return_intermediate_steps=True, 
        map_prompt=MAP_PROMPT, 
        combine_prompt=COMBINE_PROMPT,
        verbose=True,
    )

    summary = chain({"input_documents": _docs}, return_only_outputs=True)

    return summary.get('output_text', '')

@st.cache_data
def store_to_milvus(
    _docs: Document,
    collection_name: str = 'temp',
):
    init()

    MassMilvus.afrom_documents(_docs, collection_name=collection_name, connection_args=milvus_connection_args)

@st.cache_resource
def get_milvus_instance(collection_names: list[str] = []):
    init()

    embeddings = get_embedding()

    instances: list[Milvus] = []

    for cn in collection_names:
        db = Milvus(
            collection_name=cn,
            embedding_function=embeddings,
            connection_args=milvus_connection_args,
        )
        instances.append(db)

    return instances

@st.cache_data
def get_similar_documents(_instances: list[Milvus] = [], query: str = '', top_k: int = 5):
    init()

    similar_docs = []
    for instance in _instances:
        similar_docs += instance.similarity_search_with_score(query, top_k=top_k)

    # FINE-TUNING needed
    similar_docs = sorted(similar_docs, key=lambda x: float(x[1]))
    # get only the top 5
    similar_docs = similar_docs[:5]

    return similar_docs


def get_completion(prompt, engine="GPT-4", temperature=0):
    init()
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message["content"]

@st.cache_data
def rephrase_question(_query: str = '', history: str = ''):
    PROMPT = """Rephrase the following question into a declarative sentence.
    Just copy the question and paste it into the "Rephrased sentence" section if the question is already clear and short enough.    
    Don't include any punctuations in the final rephrased sentence.
    Capture the main idea of the question in the final rephrased sentence.
    Refer the following examples for the rephrasing.
    EXAMPLE:
    1. "markdown bolt" -> "markdown bolt"
    2. "how can I learn the artificial intelligence in the most efficient way?" -> "the most efficient way to learn artificial intelligence"
    3. "I wonder if Grenoble is the largest city in France?" -> "largest city France Grenoble"
    The final rephrased question should combine the current question in the "Question" section and the history of the conversation between the user and the chatbot in the "History" section.
    If the History section is empty, ignore the history's influence on the rephrased sentence.
    Make sure your sentence can be understood by a 10-year-old child.
    The final rephrased sentence should be no longer than 15 words.

    Question: {query}

    History: {history}

    Rephrased sentence:"""
    PROMPT = PROMPT.format(query=_query, history=history)
    response = get_completion(PROMPT)
    return response

@st.cache_data
def get_answer(_query: str = '', _docs: list = [Document], _chat_history: ConversationBufferMemory = ConversationBufferMemory()):
    init()

    engine = st.session_state.get("OPENAI_MODEL")
    temperature = st.session_state.get("OPENAI_TEMPERATURE")

    template = """Given the following a question from human in the "QUESTION" section and several extracted parts of a long document in the "EXTRACTED DOCUMENT" section.
    Your response should be based on the history of the conversation and the extracted parts of the document.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    QUESTION: {question}

    EXTRACTED DOCUMENT: {summaries}

    HISTORY: {chat_history}
    
    ANSWER:"""

    PROMPT = PromptTemplate(template=template, input_variables=['question', 'summaries', 'chat_history'])
    llm = get_llm(engine, temperature)
    qa_chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type='stuff',
        prompt=PROMPT,
        # memory = chat_history,
        # verbose=True,
    )

    response = qa_chain({'input_documents': _docs, 'question': _query, 'chat_history': _chat_history.buffer})

    return response

@st.cache_data
def get_answer_with_agent(_query: str = '', docs: list = [Document]):
    init()

@st.cache_data
def get_all_collections():
    init()

    connections.connect(host=MILVUS_HOST, port="19530")
    collections = utility.list_collections()
    collections = [x for x in collections if x.startswith('dev_weblink_')]
    collections.sort()
    return collections

@st.cache_data
def load_collection_from_milvus(collection_name: str = ''):
    init()

    connections.connect(host=MILVUS_HOST, port="19530")
    collection = utility.load_collection(collection_name)
    return collection

@st.cache_data
def get_collection_info(collection_name: str = '') -> dict:
    init()

    connections.connect(host=MILVUS_HOST, port='19530')
    collection = Collection(collection_name)

    field_name = ['summary', 'user_name', 'source']

    output_res: dict = {}

    for fn in field_name:
        try:
            res = collection.query(
                expr = "pk > 1",
                offset = 0,
                limit = 1, 
                output_fields = [fn],
                consistency_level="Strong"
            )
            output_res[fn] = res[0][fn]
        except exceptions.MilvusException:
            output_res[fn] =  ''

    return output_res
    
# -----FRONTEND-----
@st.cache_data
def write_history():
    messages = ss.get('chat_with_weblink_buffer', []).chat_memory.messages.copy()
    messages.reverse()
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            st.write(bot_template.format(MSG=msg.content, MODEL=ss['chat_with_weblink_cost'][i]['model'], COST=round(ss['chat_with_weblink_cost'][i]['cost'], 6), TOKENS_USED=ss['chat_with_weblink_cost'][i]['total_tokens'], PROMPT=ss['chat_with_weblink_cost'][i]['prompt'], COMPLETION=ss['chat_with_weblink_cost'][i]['completion']), unsafe_allow_html=True)
            with st.expander('Sources'):
                source = ss['chat_with_weblink_sources'][i]
                st.data_editor(pd.DataFrame(source), use_container_width=True, disabled=['Score', 'Content', 'Link', 'User'], key=f'weblink_{i}')
        else:
            st.write(user_template.format(MSG= msg.content), unsafe_allow_html=True)

def weblink_input_area():

    stage = 'dev'
    project = 'weblink'
    storage_name_prefix = f'{stage}_{project}_'

    with st.expander('Weblink resources', expanded=True):

        tab1, tab2 = st.tabs(["Your weblinks", "Shared weblinks"])
        # imported weblink by current user
        with tab1:
            st.text_input('Plug in your weblink here:', key='weblink')

            if ss.get('weblink_user_clean') and ss.get('weblink_user_list'):
                ss.get('weblink_user_list').pop(-1)

            col1, col2, _, _, _ = st.columns([1, 2, 2, 2, 3])
            with col1:
                st.button('Submit', type='primary', key='weblink_user_submit')
            with col2:
                st.button('Remove the last weblink', key='weblink_user_clean')

        # shared weblink by other users
        with tab2:
            if ss.get('weblink_db_clean') and ss.get('weblink_db_list'):
                ss.get('weblink_db_list').pop(-1)

            if not ss.get('weblink_collection_list'):
                get_all_collections.clear()
                ss['weblink_collection_list'] = get_all_collections()
            st.selectbox('Choose a collection stored in database below:', ss.get('weblink_collection_list'), key='weblink_selected_from_db')
            col1, col2, _, _, _ = st.columns([1, 2, 2, 2, 3])
            with col1:
                st.button('Add', type='primary', key='weblink_db_submit')
            with col2:
                st.button('Remove the last weblink', key='weblink_db_clean')

        if ss.get('weblink') and ss.get('weblink_user_submit'):
            urls = [ss.get('weblink', '')]

            load_urls.clear()
            split_data.clear()
            get_summary.clear()
            store_to_milvus.clear()

            doc = load_urls(urls)
            docs = split_data(doc)
            summary = get_summary(docs)

            current_user_name = ss.get('USER_INFO', {'name': ''}).get('name', '')

            for doc in docs:
                doc.metadata['summary'] = summary
                doc.metadata['user_name'] = current_user_name

            collection_name = storage_name_prefix + ss.get('weblink').split('//')[-1].replace('.', '_').replace('/', '_').replace('-', '_')

            store_to_milvus(docs, collection_name=collection_name)

            ss['weblink_selected_collections'] = ss.get('weblink_selected_collections', []) + [collection_name]

            ss['weblink_user_list'] = ss.get('weblink_user_list', []) + [{'Link': ss.get('weblink'), 'Summary': summary, 'User': current_user_name,'Not for sharing?': False}]

        if ss.get('weblink_selected_from_db') and ss.get('weblink_db_submit'):
            get_collection_info.clear()
            ss['weblink_selected_collections'] = ss.get('weblink_selected_collections', []) + [ss.get('weblink_selected_from_db')]
            info = get_collection_info(ss.get('weblink_selected_from_db'))
            ss['weblink_db_list'] = ss.get('weblink_db_list', []) + [{'Link': info.get('source', ''), 'Summary': info.get('summary', ''), 'User': info.get('user_name', '')}]


    if ss.get('weblink_user_list') or ss.get('weblink_db_list'):
        weblink_user_df = pd.DataFrame(ss.get('weblink_user_list'))
        weblink_db_df = pd.DataFrame(ss.get('weblink_db_list'))
        with st.expander(label='Weblinks selected for the chat', expanded=True):
            st.write('Links imported by you:') if ss.get('weblink_user_list') else None
            weblink_display = st.data_editor(weblink_user_df, use_container_width=True, disabled=['Link', 'Summary']) if ss.get('weblink_user_list') else None
            st.write('Links shared by others:') if ss.get('weblink_db_list') else None
            st.data_editor(weblink_db_df, use_container_width=True, disabled=['Link', 'Summary']) if ss.get('weblink_db_list') else None

def initialize_session_state():
    ss.pop('chat_with_weblink_buffer', None)
    ss.pop('chat_with_weblink_cost', None)
    ss.pop('chat_with_weblink_sources', None)

def qa_area():
    st.write(css, unsafe_allow_html=True)

    if ss.get('weblink_user_list') or ss.get('weblink_db_list'):
        st.text_input('Plug in your question here', key='chat_with_weblink_q')
        submit_col, clear_col, _, _, _ = st.columns([1, 1, 3, 3, 3])
        with submit_col:
            button_submit = st.button('Submit', type="primary")
        with clear_col:
            button_clear = st.button('Clear')

        if ss.get('chat_with_weblink_q') and button_submit:
            
            rephrase_question.clear()
            get_milvus_instance.clear()
            get_similar_documents.clear()
            get_answer.clear()
            write_history.clear()

            current_q = ss.get('chat_with_weblink_q')

            chat_history = ss.get('chat_with_weblink_buffer', ConversationBufferMemory())
            chat_history_string : str = chat_history.buffer

            rephrased_q = rephrase_question(current_q, chat_history_string)
            st.write(f'Searching for: **{rephrased_q}**')

            collection_names = ss.get('weblink_selected_collections', [])
            instances = get_milvus_instance(collection_names)
            similar_docs_with_score = get_similar_documents(_instances=instances, query=rephrased_q)
            similar_docs = [x[0] for x in similar_docs_with_score]

            with get_openai_callback() as cb:
                response = get_answer(_query=current_q, _docs=similar_docs, _chat_history=chat_history)

            chat_history.save_context({'input': current_q}, {'output': response['output_text']})
            ss['chat_with_weblink_buffer'] = chat_history

            current_source = [{
                'Score': round(doc[1], 5),
                'Content': doc[0].page_content,
                'Link': doc[0].metadata['source'],
                'User': doc[0].metadata.get('user_name', ''),
            } for doc in similar_docs_with_score]
            ss['chat_with_weblink_sources'] = [current_source] + [[{}]] + ss.get('chat_with_weblink_sources', [])

            ss['chat_with_weblink_cost'] = [get_cost_dict(cb)] + [{}] + ss.get('chat_with_weblink_cost', [])

        if button_clear:
            initialize_session_state()

    write_history()
