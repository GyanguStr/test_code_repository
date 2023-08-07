from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import os
import streamlit as st
from modules.model import get_llm

def create_agent(filename: str):

    os.environ["OPENAI_API_KEY"] = st.session_state.get("OPENAI_API_KEY")
    engine = st.session_state.get("OPENAI_MODEL")
    temperature = st.session_state.get("OPENAI_TEMPERATURE")
    os.environ["OPENAI_API_BASE"] = 'https://mobile-beacon.openai.azure.com/'
    os.environ['OPENAI_API_VERSION'] = '2023-05-15'
    os.environ['OPENAI_API_TYPE'] = 'azure'

    llm = get_llm(engine, temperature)
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(llm, df, verbose=True)

def query_agent(agent, query):

    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

    response = agent.run(prompt)

    return response.__str__()
