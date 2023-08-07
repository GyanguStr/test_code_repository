import os
import datetime
import streamlit as st
from components.html_templates import user_template, old_bot_template
from langchain.schema import SystemMessage, HumanMessage, AIMessage, Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from modules.model import get_llm
from langchain.callbacks import get_openai_callback
from components.frontend_cost import get_cost_dict
from modules.conversation_control import PostgreSQLConversationTracker, MilvusDBConversationTracker
from modules.mass_milvus import MassMilvus
import pandas as pd

ss = st.session_state

# --- BACKEND ---
def init():
    os.environ["OPENAI_API_KEY"] = ss.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
    os.environ['OPENAI_API_VERSION'] = os.getenv("OPENAI_API_VERSION")
    os.environ['OPENAI_API_TYPE'] = 'azure'
    global engine
    global temperature
    engine = ss.get("OPENAI_MODEL")
    temperature = ss.get("OPENAI_TEMPERATURE")

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

def initialize_buffer(buffer: str):
    if buffer not in ss:
        ss[buffer] = ConversationBufferMemory()

@st.cache_data
def get_answer(query: str, buffer: str, cost: str):
    init()

    llm = get_llm(model=engine, temperature=temperature)
    conversation_chain = ConversationChain(llm=llm, memory=ss[buffer])

    with get_openai_callback() as cb:
        result = conversation_chain.run(query)

    ss[cost] = [get_cost_dict(cb)] + [{}] + ss.get(cost, [])
    return result

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
    4. "Who is Joe Biden" -> "Joe Biden"
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
def get_pg_id(id_name):
    pgct = PostgreSQLConversationTracker()
    try:
        id = pgct.store_to_conversation_tracker(ss['USER_INFO']['name'])
        ss[id_name] = id
    except Exception as e:
        raise e

def create_tacker_document(rephrased_q: str, q: str, a: str, id: int, private: bool):
    created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "question": q,
        "answer": a,
        "conversation_tacker_id": id,
        'private': private,
        "created_at": created_time,
    }
    return Document(page_content=rephrased_q, metadata=metadata)

@st.cache_data
def store_history_to_milvus(_doc: Document, collection_name: str):
    MILVUS_HOST='52.226.226.29'
    milvus_connection_args = {'host': MILVUS_HOST}
    pk = MassMilvus.afrom_documents(documents=[_doc], collection_name=collection_name, connection_args=milvus_connection_args)
    return pk

@st.cache_data
def get_history_for_user(user_name: str, return_limit: int = 100):
    pgct = PostgreSQLConversationTracker()
    mct = MilvusDBConversationTracker()
    ids = pgct.get_conversation_tracker_id(user_name, return_limit)
    ids.reverse()
    history = mct.get_conversation_history(ids)
    history = sorted(history, key=lambda x: x['created_at'], reverse=True)
    ss['raw_user_history'] = history
    ss['user_history_summary_id'] = []
    ss['user_history_summary'] = []
    count = 0
    for i, h in enumerate(history):
        current_id = h['conversation_tacker_id']
        if i != 0:
            if previous_id == current_id:
                continue
            else:
                previous_id = current_id
        else:
            previous_id = current_id
        ss['user_history_summary'].append(f"[{count}] Topic: {h['text']} --- Time: {h['created_at']}")
        ss['user_history_summary_id'].append(h['conversation_tacker_id'])
        count += 1

def resume_user_history(buffer, cost):
    index = int(ss.get('selected_user_history').split(']')[0].replace('[', ''))
    id = ss.get('user_history_summary_id')[index]
    history = ss.get('raw_user_history')
    filtered_history = [h for h in history if h['conversation_tacker_id'] == id]
    filtered_history = sorted(filtered_history, key=lambda x: x['created_at'])
    # initialize buffer, cost and pg_id
    ss[buffer] = ConversationBufferMemory()
    ss[cost] = []
    ss['pg_id'] = id
    cost_template = {
        'model': '-',
        'cost': 0,
        'total_tokens': 0,
        'prompt': 0,
        'completion': 0,
    }
    for h in filtered_history:
        ss[buffer].save_context({'input': h['question']}, {'output': h['answer']})
        ss[cost] = [cost_template] + [{}] + ss.get(cost, [])

@st.cache_data
def search_similar_answer_in_db(rephrased_q: str, query: str, buffer: str, cost: str,similar: str):
    mct = MilvusDBConversationTracker()
    similar_response, similar_cost = mct.get_similar_qa(rephrased_q)
    if similar_response:
        ss[buffer].save_context({'input': query}, {'output': similar_response})
        ss[cost] = [similar_cost] + [{}] + ss.get(cost, [])
        ss[similar] = True
        return True
    else:
        return False


# --- FRONTEND ---
def clear_submit():
    ss["submit"] = False

def write_history(buffer: list, cost: list, similar: bool = False):
    messages = buffer.chat_memory.messages.copy()
    messages.reverse()
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            st.write(old_bot_template.format(MSG=msg.content, MODEL=cost[i]['model'], COST=round(cost[i]['cost'], 6), TOKENS_USED=cost[i]['total_tokens'], PROMPT=cost[i]['prompt'], COMPLETION=cost[i]['completion']), unsafe_allow_html=True)
            if similar and i == 2:
                thumbs_up, thumbs_down, _, _, _ = st.columns([0.5, 0.5, 2, 2, 2])
                with thumbs_up:
                    thumbs_up_button = st.button('👍', key='thumbs_up')
                with thumbs_down:
                    thumbs_down_button = st.button('👎', key='thumbs_down')
        else:
            st.write(user_template.format(MSG=msg.content), unsafe_allow_html=True)

def initialize_session_state(pop_keys: list = []):
    for key in pop_keys:
        ss.pop(key, None)

def qa_area():
    buffer_name = 'history_concept_buffer'
    cost_name = 'history_concept_cost'
    conversation_tracker_db_name = 'conversation_tracker'
    pg_id_name = 'pg_id'
    similar_a_name = 'similar_a'

    initialize_buffer(buffer_name)
    get_pg_id.clear()
    if pg_id_name not in ss:
        get_pg_id(pg_id_name)

    with st.expander('Your chat history'):
        if not ss.get('user_history_summary'):
            get_history_for_user.clear()
            get_history_for_user(ss['USER_INFO']['name'])
        st.radio('Which history do you want to resume?', ss['user_history_summary'], key='selected_user_history')
        
        resume_col, reload_col, _, _, _ = st.columns([1, 1, 1, 2, 2])
        with resume_col:
            st.button('Resume', key='resume_user_history', type='primary')
        with reload_col:
            st.button('Reload', key='reload_user_history')
        
        if ss.get('resume_user_history'):
            resume_user_history(buffer_name, cost_name)
        
        if ss.get('reload_user_history'):
            get_history_for_user.clear()
            get_history_for_user(ss['USER_INFO']['name'])

    st.text_area('question', key='question', height=100, placeholder='Enter question here', help='',
                 label_visibility="collapsed", on_change=clear_submit())

    submit_col, clear_col, mode_col, _, _ = st.columns([1, 1, 2, 2, 2])
    with submit_col:
        button_submit = st.button('Submit', type="primary")
    with clear_col:
        button_clear = st.button('Clear')
    with mode_col:
        st.checkbox('Private mode', key='private_mode')

    if button_clear:
        initialize_session_state([buffer_name, cost_name])
        get_pg_id.clear()
        get_pg_id(pg_id_name)
        ss["submit"] = False

    if ss.get('question') and button_submit:
        
        ss[similar_a_name] = False

        if not ss.get("OPENAI_API_KEY"):
            st.error("Please configure your Azure OpenAI API key!")
        else:
            rephrase_question.clear()
            get_answer.clear()
            store_history_to_milvus.clear()
            search_similar_answer_in_db.clear()

            prompt = ss.get("question")
            rephrased_q = rephrase_question(prompt, ss[buffer_name].buffer)
            st.write(f'Searching for: **{rephrased_q}**')

            show_similar_answer = search_similar_answer_in_db(rephrased_q, prompt, buffer_name, cost_name, similar_a_name)

            if not show_similar_answer:
                result = get_answer(prompt, buffer_name, cost_name)
                doc = create_tacker_document(rephrased_q, prompt, result, ss[pg_id_name], ss['private_mode'])
                pk = store_history_to_milvus(doc, conversation_tracker_db_name)
                print('-----', pk)
            

    if ss.get(buffer_name) and ss.get(cost_name):
        cost = ss[cost_name]
        buffer = ss[buffer_name]
        write_history(buffer, cost, ss.get(similar_a_name))
    
    

    # st.write(ss)