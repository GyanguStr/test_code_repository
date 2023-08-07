import streamlit as st
from components.sidebar import sidebar
from modules.chat import chat_with_history
from components.html_templates import user_template, bot_template, css
from components.authentication import AzureAuthentication
from modules.chat_with_json import RenderPage, ChatService

ss = st.session_state

st.set_page_config(
    page_title="Figma component identifier",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.write(css, unsafe_allow_html=True)


def display_page():
    model_options = ('GPT-3-16K', 'GPT-4', 'GPT-4-32K')
    sidebar(model_options)
    st.write(css, unsafe_allow_html=True)
    st.title('Figma component identifier')
    if not ss.get("OPENAI_API_KEY"):
        st.error("Please configure your Azure OpenAI API key!")
    else:
        ChatService()
        with st.spinner(":green[Loading...]"):
            uploaded_file = st.file_uploader("**:point_right: Choose a Json file**", type=["json"])
            json_data = RenderPage.get_uploaded_file(uploaded_file)
            ss["json_uploaded_file"] = json_data
        RenderPage.get_question()
        RenderPage.get_chat_record()
        # st.write(st.session_state)


AzureAuthentication.check_token(display_page)