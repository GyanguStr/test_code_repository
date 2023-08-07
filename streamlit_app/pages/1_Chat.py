import streamlit as st
from components.sidebar import sidebar
from modules.chat import single_chat
from components.authentication import AzureAuthentication

st.set_page_config(
    page_title="Chat",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def display_page():
    st.title('Chat')

    sidebar()

    prompt = st.text_input('Plug in your prompt here')

    if prompt:

        if not st.session_state.get("OPENAI_API_KEY"):
            st.error("Please configure your Azure OpenAI API key!")
        else:
            response = single_chat(prompt)

            st.write(st.session_state["single_chat_buffer"][-1])


AzureAuthentication.check_token(display_page)
