import streamlit as st
from components.sidebar import sidebar
from components.html_templates import css
from components.authentication import AzureAuthentication
from modules.chat_with_memory import qa_area

ss = st.session_state

st.set_page_config(
    page_title="Chat with memory (knowledge sharing)",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.write(css, unsafe_allow_html=True)

st.title('Chat with memory (knowledge sharing)')

def display():
    model_options = ('GPT-3', 'GPT-4', 'GPT-4-32K')
    sidebar(model_options)

    qa_area()


AzureAuthentication.check_token(display)
