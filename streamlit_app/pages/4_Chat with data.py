import streamlit as st
import pandas as pd
import json
from components.sidebar import sidebar
from modules.chat import chat_with_database
from modules.agent import query_agent, create_agent
from components.authentication import AzureAuthentication

ss = st.session_state

st.set_page_config(
    page_title="Chat with data",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def decode_response(response: str) -> dict:
    return json.loads(response)


def write_response(response_dict: dict):
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


def main():

    sidebar()

    data_source = st.selectbox(
        'Data source',
        ('database', 'csv files')
    )

    if data_source == 'database':

        st.title('Chat with database')

        database_name = st.selectbox(
            'Choose a database',
            ('Fake bank app size',)
        )

        prompt = st.text_input('What do you want to know about this database?')

        if prompt:
            if not st.session_state.get("OPENAI_API_KEY"):
                st.error("Please configure your Azure OpenAI API key!")
            else:
                respnse = chat_with_database(prompt, database_name)

                st.write('🤖', respnse['result'])

    elif data_source == 'csv files':
        st.title("Chat with csv files")

        st.write("Please upload your CSV file below.")

        data = st.file_uploader("Upload a CSV")

        query = st.text_area("Insert your query")

        if st.button("Submit", type="primary"):
            # Create an agent from the CSV file.
            agent = create_agent(data)

            # Query the agent.
            response = query_agent(agent=agent, query=query)

            # Decode the response.
            decoded_response = decode_response(response)

            # Write the response to the Streamlit app.
            write_response(decoded_response)


AzureAuthentication.check_token(main)