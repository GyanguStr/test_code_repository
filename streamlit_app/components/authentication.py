import os
import json
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Callable
from msal_streamlit_authentication import msal_authentication
import extra_streamlit_components as stx
from cryptography.fernet import Fernet


@dataclass
class AzureAuthentication:
    """class that verifies logged-in users and obtains user information through Azure AD"""

    load_dotenv(verbose=True)

    client_id: str = (
        os.getenv("CLIENT_ID") if os.getenv("CLIENT_ID") else ""
    )
    tenant_id: str = (
        os.getenv("TENANT_ID") if os.getenv("TENANT_ID") else ""
    )
    redirect_uri: str = (
        os.getenv("REDIRECT_URI") if os.getenv("REDIRECT_URI") else "http://localhost:8501"
    )

    @staticmethod
    def get_token() -> dict:
        value = msal_authentication(
            auth={
                "clientId": AzureAuthentication.client_id,
                "authority": f"https://login.microsoftonline.com/{AzureAuthentication.tenant_id}",
                "redirectUri": AzureAuthentication.redirect_uri,
                "postLogoutRedirectUri": AzureAuthentication.redirect_uri
            },
            cache={
                "cacheLocation": "sessionStorage",
                "storeAuthStateInCookie": False
            },
            login_request={
                "scopes": ["https://graph.microsoft.com/User.ReadBasic.All"]
            },
            key=1)
        if not value:
            cookie_manager = stx.CookieManager()
            try:
                cookie_manager.delete("user_info")
            except KeyError:
                pass
        return value

    @staticmethod
    def generate_cookie(value: str):
        private_key = Fernet.generate_key()
        f = Fernet(key=private_key)
        value = value.encode(encoding="utf-8")
        user_info = f.encrypt(value)
        user_info = user_info.decode()
        private_key = private_key.decode()
        cookie = f"{private_key}|beacon|{user_info}"
        return cookie

    @staticmethod
    def check_token(page_func: Callable):
        cookie_manager = stx.CookieManager()
        cookie = cookie_manager.get(cookie='user_info')
        if not cookie:
            st.error("Please log in and try again")
        else:
            private_key, user_info = cookie.split("|beacon|")
            private_key = private_key.encode(encoding="utf-8")
            user_info = user_info.encode(encoding="utf-8")
            f = Fernet(private_key)
            user_info = json.loads(f.decrypt(user_info).decode())
            if AzureAuthentication.tenant_id == user_info["tenantId"]:
                st.session_state["USER_INFO"] = user_info
                page_func()
            else:
                st.error("Authentication information does not match, please log in again")
