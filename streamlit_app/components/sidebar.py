import streamlit as st
from components.key_vault import FetchKey


def set_openai_api_key():
    st.session_state["OPENAI_API_KEY"] = FetchKey("OPENAI-KEY").retrieve_secret()


def set_openai_model(model: str):
    st.session_state["OPENAI_MODEL"] = model


def set_openai_temperature(temperature: str):
    st.session_state["OPENAI_TEMPERATURE"] = temperature


def set_openai_embedding_model(embedding_model: str):
    st.session_state["OPENAI_EMBEDDING_MODEL"] = embedding_model


def sidebar(model_options=None):
    model_options = model_options if model_options else ('Davinci', 'GPT-3', 'GPT-4', 'GPT-4-32K')

    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Choose the model as you want 🤖\n"
            "2. Tune the temperature as you want 🌡️\n"
            "3. Have fun\n"
        )

        model = st.selectbox(
            'Model',
            model_options
        )

        temperature = st.slider(
            'Temperature',
            0.0, 1.0, (0.2),
            step=0.1)
        with st.expander("Advanced"):
            embedding_model = st.selectbox(
                'Embedding model',
                ('Ada-embedding',)
            )

        set_openai_api_key()

        if model:
            set_openai_model(model)

        if embedding_model:
            set_openai_embedding_model(embedding_model)

        if temperature:
            set_openai_temperature(temperature)
        else:
            set_openai_temperature(0.0)

        st.markdown("---")
