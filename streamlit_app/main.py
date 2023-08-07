from components.html_templates import hide_bar
from components.authentication import AzureAuthentication
import streamlit as st
import extra_streamlit_components as stx
import datetime
import json

st.set_page_config(
    page_title="Main",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('🦜🔗 Useful concepts')

login_token = AzureAuthentication.get_token()


if not login_token:
    st.warning("Welcome to beacon cognitive search, Please click the login button to log in...")
    st.markdown(hide_bar, unsafe_allow_html=True)

if login_token:
    account = login_token["account"]
    account_json = json.dumps(account)
    user_info = AzureAuthentication.generate_cookie(account_json)
    cookie_manager = stx.CookieManager()
    cookie_manager.set(cookie="user_info", val=user_info,
                       expires_at=datetime.datetime.today() + datetime.timedelta(days=1))
    username = login_token["account"]["name"]
    st.sidebar.title(f"Welcome {username}")
    st.sidebar.success("Select a module above")
    st.warning("If you want to log out, please click the logout button above...")
