import os
import streamlit as st
from components.sidebar import sidebar
from modules.chat import chat_with_history
from components.html_templates import user_template, bot_template, css
from components.authentication import AzureAuthentication
import datetime

ss = st.session_state

st.set_page_config(
    page_title="Chat with memory",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.write(css, unsafe_allow_html=True)


def clear_submit():
    ss["submit"] = False


def write_history():
    messages = ss.get('chat_with_history_buffer', []).copy()

    messages.reverse()

    for i, msg in enumerate(messages[:-1]):
        if i % 2 == 0:
            st.write(bot_template.format(MSG=msg.content,
                                         MODEL=ss['chat_with_history_cost'][i]['model'],
                                         COST=round(ss['chat_with_history_cost'][i]['cost'], 6),
                                         TOKENS_USED=ss['chat_with_history_cost'][i]['total_tokens'],
                                         PROMPT=ss['chat_with_history_cost'][i]['prompt'],
                                         COMPLETION=ss['chat_with_history_cost'][i]['completion'],
                                         TIME=ss['time_spent_chat_with_history'][i]),
                     unsafe_allow_html=True)
        else:
            st.write(user_template.format(MSG=msg.content), unsafe_allow_html=True)


def initialize_session_state():
    ss.pop('chat_with_history_buffer', None)
    ss.pop('chat_with_history_cost', None)
    ss.pop('time_spent_chat_with_history', None)
    ss['conversation_id'] = datetime.datetime.now()


def main():
    st.title('Chat with memory')
    if ss.get('conversation_id') is None:
        ss['conversation_id'] = datetime.datetime.now()

    model_options = ('GPT-3', 'GPT-4', 'GPT-4-32K')

    sidebar(model_options)

    st.text_area('question', key='question', height=100, placeholder='Enter question here', help='',
                 label_visibility="collapsed", on_change=clear_submit())

    submit_col, clear_col, _, _, _ = st.columns([1, 1, 2, 2, 2])

    with submit_col:
        button_submit = st.button('Submit', type="primary")

    with clear_col:
        button_clear = st.button('Clear')

    if button_clear:
        initialize_session_state()
        ss["submit"] = False
    if button_submit or ss.get("submit"):
        if not ss.get("OPENAI_API_KEY"):
            st.error("Please configure your Azure OpenAI API key!")
        else:
            ss["submit"] = True
            # Find the chat with history function that you want to check in streamlit_app/modules/chat.py
            # change the function name to chat_with_history
            prompt = ss.get("question")
            chat_with_history(prompt)

    if ss.get('chat_with_history_buffer') and ss.get('chat_with_history_cost'):
        write_history()
    # st.write(ss)



AzureAuthentication.check_token(main)
