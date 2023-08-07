import streamlit as st
from components.authentication import AzureAuthentication
from modules.chat_with_roadmap import roadmap_input_area, roadmap_qa_area, roadmap
from components.html_templates import css

ss = st.session_state

st.set_page_config(
    page_title="Chat with roadmap",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.write(css, unsafe_allow_html=True)

st.title("Chat with roadmap")

def main():
    roadmap_input_area()
    roadmap_qa_area()
    roadmap()

AzureAuthentication.check_token(main)

# ss
