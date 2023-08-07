from components.sidebar import sidebar
import streamlit as st
from langchain.memory import ConversationBufferMemory
from components.html_templates import user_template, bot_template, css
from modules.storage import get_all_collections, get_slack_docs, store_to_milvus_vector_database, get_milvus_instance
from modules.chat_with_slack import chat_with_slack
from components.authentication import AzureAuthentication
import pandas as pd

ss = st.session_state

st.set_page_config(
    page_title="Chat with Slack",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clear_submit():
    st.session_state["submit"] = False


def initialize_session_state():
    ss.pop('chat_with_slack_buffer', None)
    ss.pop('chat_with_slack_cost', None)
    ss.pop('chat_with_slack_sources', None)


def write_history():
    messages = ss.get('chat_with_slack_buffer', []).chat_memory.messages.copy()
    messages.reverse()
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            st.write(bot_template.format(MSG=msg.content, MODEL=ss['chat_with_slack_cost'][i]['model'], COST=round(ss['chat_with_slack_cost'][i]['cost'], 6), TOKENS_USED=ss['chat_with_slack_cost'][i]['total_tokens'], PROMPT=ss['chat_with_slack_cost'][i]['prompt'], COMPLETION=ss['chat_with_slack_cost'][i]['completion']), unsafe_allow_html=True)
            score = []
            content = []
            user = []
            channel = []
            timestamp = []
            chunk = []
            with st.expander('Sources'):
                source = ss['chat_with_slack_sources'][i]
                for s in source:
                    
                    score.append(s['score'])
                    content.append(s['content'])
                    user.append(s['user'])
                    channel.append(s['channel'])
                    timestamp.append(s['timestamp'])
                    chunk.append(s['chunk'])
                st.table(pd.DataFrame({'score': score, 'content': content, 'user': user, 'channel': channel, 'timestamp': timestamp, 'chunk': chunk}))
                    
                # for src in source:
                #     for s in src.split('\n'):
                #         st.markdown(s)
        else:
            st.write(user_template.format(MSG= msg.content), unsafe_allow_html=True)


def file_loader():
    with st.expander('Upload a file'):
        st.text_input('Enter a name for the file', key='file_name',
                      help='It is preferable to use the URL of the slack workspace, e.g. https://myworkspace.slack.com')
        st.file_uploader('Upload a file', type='zip', key='zip_file', label_visibility="collapsed")

        file_name = ss.get('file_name')
        zip_file = ss.get('zip_file')
        model = ss.get('OPENAI_MODEL')
        embedding_model = ss.get('OPENAI_EMBEDDING_MODEL', 'Ada-embedding')

        disabled = not (file_name and zip_file and model)

        st.button('Upload', key='upload', disabled=disabled)

        if ss.get('upload'):
            doc_chunks = get_slack_docs(zip_file, file_name)
            store_to_milvus_vector_database(doc_chunks, embedding_model, file_name)
            st.success('File uploaded successfully!')
            ss['collection_list'] = get_all_collections()

    st.selectbox('Select a collection', ss.get('collection_list'), key='selected_collection',
                 label_visibility="collapsed")

def get_completion(input, engine="gpt-4", temperature=0, detailed=False, one_shot=True): 
    import openai
    from components.key_vault import FetchKey

    openai.api_key = FetchKey("OPENAI-KEY").retrieve_secret()

    if one_shot:
        messages = [{"role": "user", "content": input}]
    else:
        messages = input

    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=temperature, 
    )

    if detailed:
        return response
    else:
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


def ask_question():
    if not ss.get('chat_with_slack_buffer'):
        ss['chat_with_slack_buffer'] = ConversationBufferMemory()

    question = st.text_area('question', key='question', height=100, placeholder='Enter question here', help='',
                            label_visibility="collapsed", on_change=clear_submit())

    submit_col, clear_col, _, _, _ = st.columns([1, 1, 2, 2, 2])
    with submit_col:
        button_submit = st.button('Submit', type="primary")
    with clear_col:
        button_clear = st.button('Clear')

    if button_clear:
        initialize_session_state()
        ss["submit"] = False

    if button_submit or st.session_state.get("submit"):
        ss["submit"] = True
        rephrased_q = rephrase_question(question, ss['chat_with_slack_buffer'].buffer)
        st.write(f'Searching for: **{rephrased_q}**')
        response = chat_with_slack(rephrased_q)

# Entry function
def display_page():
    model_options = ('Davinci', 'GPT-3', 'GPT-4', 'GPT-4-32K')
    sidebar(model_options)
    st.write(css, unsafe_allow_html=True)
    st.title('Chat with Slack')
    if not ss.get("OPENAI_API_KEY"):
        st.error("Please configure your Azure OpenAI API key!")
    else:
        ss['collection_list'] = get_all_collections()
        file_loader()
        get_milvus_instance()
        ask_question()

    b = ss.get('chat_with_slack_buffer')
    if b:
        try:
            write_history()
        except:
            pass
    # st.write(ss)


AzureAuthentication.check_token(display_page)
