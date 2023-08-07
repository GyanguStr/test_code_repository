import streamlit as st
from modules.chat_with_weblink import weblink_input_area, qa_area
from components.authentication import AzureAuthentication
from components.sidebar import sidebar

ss = st.session_state

st.set_page_config(
    page_title="Chat with weblink",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def display_page():
    st.title("Chat with Weblink")
    model_options = ('GPT-3', 'GPT-4', 'GPT-4-32K')
    sidebar(model_options)
    weblink_input_area()
    qa_area()
    # ss


AzureAuthentication.check_token(display_page)
