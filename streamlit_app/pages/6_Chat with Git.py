from components.sidebar import sidebar
import streamlit as st
from components.html_templates import user_template, bot_template, css
from modules.chat_with_git import RenderPage, ChatService
from components.authentication import AzureAuthentication

ss = st.session_state

st.set_page_config(
    page_title="Chat with Git",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write(css, unsafe_allow_html=True)


def display_page():
    model_options = ('GPT-4', 'GPT-4-32K')
    sidebar(model_options)
    st.write(css, unsafe_allow_html=True)
    if not ss.get("OPENAI_API_KEY"):
        st.error("Please configure your Azure OpenAI API key!")
    else:
        ChatService()
        milvus_host = '52.226.226.29'
        RenderPage.get_collections(database_host=milvus_host)
        _, col1, _, col2 = st.columns([1, 3, 1, 2])
        with col1:
            st.title('Chat with Git')
            st.selectbox('Select a repo collection to chat',
                         ss.get('collection_git_list'),
                         key='selected_git_collection')
            instance_list = ChatService.get_milvus_instance(collection_name=ss.get("selected_git_collection"))
            ss['milvus_git_instance'] = instance_list
            RenderPage.get_question()
            RenderPage.get_chat_record()
        with col2:
            RenderPage.display_conversation_history()
        # st.write(st.session_state)


AzureAuthentication.check_token(display_page)
