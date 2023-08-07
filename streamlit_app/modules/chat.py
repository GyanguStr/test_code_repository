from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain import SQLDatabase
from modules.model import model_dict
from components.postgre_wrapper import PgService
# Chat with history
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from modules.model import get_llm
import streamlit as st
from modules.model import model_dict, get_llm, chat_openai
from components.frontend_cost import get_cost_dict, timer_decorator
import os

global engine
global temperature


def init():
    os.environ["OPENAI_API_KEY"] = st.session_state.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
    os.environ['OPENAI_API_VERSION'] = os.getenv("OPENAI_API_VERSION")
    os.environ['OPENAI_API_TYPE'] = 'azure'
    global engine
    global temperature
    engine = st.session_state.get("OPENAI_MODEL")
    temperature = st.session_state.get("OPENAI_TEMPERATURE")


database_dict = {
    'Fake bank app size': 'app_size'
}

pg_service = PgService()


@pg_service.insert_decorator
def single_chat(input):
    init()

    query = [
        SystemMessage(content="You are a nice AI bot that can answer any question in the world."),
        HumanMessage(content=input),
    ]

    llm = get_llm(model=engine, temperature=temperature)

    with get_openai_callback() as cb:
        if engine == 'GPT-4' or engine == 'GPT-3' or engine == 'GPT-4-32K':
            response = llm(query).content
        else:
            response = llm(input)
        print(cb)
    st.session_state["single_chat_buffer"] = query
    st.session_state["single_chat_buffer"].append(response)
    return cb


def chat_with_database(input, database_name):
    init()

    llm = get_llm(engine, temperature)

    PG_USER = os.environ.get('PG_USER')
    PG_PASSWORD = os.environ.get('PG_PASSWORD')

    pg_connection_string = f'postgresql+psycopg2://citus:Beacon2023@c.gpt-project-cosmos-db-postgresql.postgres.database.azure.com:5432/citus'

    db = SQLDatabase.from_uri(pg_connection_string, include_tables=[database_dict[database_name]])

    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    response = db_chain(input)

    return response


def chat_with_history_01(input):
    """
    *** NOT IN USE ***
    test with ConversationChain
    """

    init()

    # initialize memory
    try:
        memory = st.session_state['chat_history']
    except KeyError:
        memory = ConversationBufferMemory()
        st.session_state['chat_history'] = memory

    llm = AzureOpenAI(
        temperature=temperature,
        deployment_name=model_dict[engine]['deployment_name'],
        model=model_dict[engine]['model_name'],
    )

    memory = st.session_state['chat_history']

    conversation_buf = ConversationChain(
        memory=memory,
        llm=llm,
        verbose=True,
    )

    response = conversation_buf(input)

    st.session_state['chat_history'] = memory

    # print response to the frontend
    st.write(response)

    return response


def chat_with_history_02(input):
    """
    *** NOT IN USE ***
    test with LLMChain
    """

    init()

    template = """
    You are a helpful chatbot. Answer the human's question below 

    Human: {human_input}
    """


# @pg_service.insert_decorator
@timer_decorator
def chat_with_history(input):
    """
    *** IN USE ***
    test with ChatOpenAI
    """

    init()

    # chat = ChatOpenAI(
    #     temperature=temperature,
    #     engine=model_dict[engine]['deployment_name'],
    # )
    chat = get_llm(model=engine, temperature=temperature)
    # initialize memory
    system_message = "You are a helpful assistant. Answer the human's question."

    if 'chat_with_history_buffer' not in st.session_state:
        st.session_state['chat_with_history_buffer'] = [SystemMessage(content=system_message)]

    st.session_state['chat_with_history_buffer'].append(HumanMessage(content=input))

    with get_openai_callback() as cb:
        response = chat(st.session_state['chat_with_history_buffer'])
    st.session_state['chat_with_history_cost'] = [get_cost_dict(cb)] + [{}] + st.session_state.get('chat_with_history_cost', [])
    st.session_state['chat_with_history_buffer'].append(AIMessage(content=response.content))
    return cb
