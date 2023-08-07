import os
import json
import pandas as pd
import streamlit as st
from json import JSONDecodeError
from typing import IO
from fuzzywuzzy import fuzz
from dataclasses import dataclass, field
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Milvus
from components.chat_prompt import CustomPromptTemplate, CustomOutputParser
from langchain.callbacks import get_openai_callback, StreamlitCallbackHandler
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from modules.model import get_llm, model_dict, get_embedding
from components.chat_prompt import ChatPrompt
from langchain.base_language import BaseLanguageModel
from components.frontend_cost import timer_decorator, get_cost_dict
from components.count_tokens import CountTokens, keys_to_extract
from components.html_templates import cost_template

ss = st.session_state


def clear_submit():
    st.session_state["submit"] = False


def initialize_session_state():
    ss.pop('chat_with_json_buffer', None)
    ss.pop('chat_with_json_cost', None)
    ss.pop('time_spent_chat_with_json', None)
    ss.pop('json_uploaded_file', None)


class RenderPage:
    """This class defines the functions required by the page"""

    @staticmethod
    def get_question():
        uploaded_file = ss.get("json_uploaded_file")
        question = st.text_area('question', key='question', height=100, placeholder='Enter question here', help='',
                                label_visibility="collapsed", on_change=clear_submit())
        submit_col, clear_col, _, _, _, regenerate_col = st.columns([2, 2, 3, 2, 3, 3])
        with submit_col:
            button_submit = st.button('Submit', type="primary")
        with clear_col:
            button_clear = st.button('Clear')
        # with regenerate_col:
        #     button_regenerate = st.button("Regenerate")

        if button_clear:
            initialize_session_state()
            ss["submit"] = False

        if button_submit or st.session_state.get("submit"):
            ss["submit"] = True
            if not uploaded_file:
                st.warning(":exclamation: You must first upload a Json file")
                return
            if question:
                with st.spinner(":green[Generating...]"):
                    ChatService.chat_with_json(question, uploaded_file=uploaded_file)

    @staticmethod
    @st.cache_data
    def get_uploaded_file(uploaded_file: IO):
        if uploaded_file is not None:
            json_data = json.load(uploaded_file)
        elif ss.get("json_uploaded_file"):
            json_data = ss.get("json_uploaded_file")
        else:
            return None
        return json_data

    @staticmethod
    def get_chat_record():
        messages = ss.get('chat_with_json_buffer', ConversationBufferMemory()).chat_memory.messages.copy()
        messages.reverse()
        for i, msg in enumerate(messages):
            if i % 2 == 0:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict):
                        st.chat_message("assistant").data_editor(pd.DataFrame([data]),
                                                                 hide_index=True,
                                                                 disabled=True,
                                                                 use_container_width=True
                                                                 )
                    if isinstance(data, list):
                        st.chat_message("assistant").data_editor(pd.DataFrame(data),
                                                                 hide_index=True,
                                                                 disabled=True,
                                                                 use_container_width=True
                                                                 )
                except JSONDecodeError:
                    st.chat_message("assistant").write(msg.content)
                st.write(cost_template.
                         format(MODEL=ss['chat_with_json_cost'][i]['model'],
                                COST=round(ss['chat_with_json_cost'][i]['cost'], 6),
                                TOKENS_USED=ss['chat_with_json_cost'][i]['total_tokens'],
                                PROMPT=ss['chat_with_json_cost'][i]['prompt'],
                                COMPLETION=ss['chat_with_json_cost'][i]['completion'],
                                TIME=ss['time_spent_chat_with_json'][i]),
                         unsafe_allow_html=True)
            else:
                st.chat_message("user").write(msg.content)
                st.write("---")


@dataclass
class ChatService:
    """This class defines the chat service by the page"""

    api_type: str = "azure"
    api_base: str = os.getenv('OPENAI_API_BASE')
    api_version: str = os.getenv("OPENAI_API_VERSION")
    api_key: str = field(init=False)

    def __post_init__(self):
        if st.session_state.get("OPENAI_API_KEY"):
            self.api_key = ss.get("OPENAI_API_KEY")
            os.environ['OPENAI_API_TYPE'] = self.api_type
            os.environ["OPENAI_API_BASE"] = self.api_base
            os.environ['OPENAI_API_VERSION'] = self.api_version
            os.environ["OPENAI_API_KEY"] = self.api_key
        else:
            raise "NoTokenError"

    @staticmethod
    def get_summer(query: str, llm: BaseLanguageModel):
        history = ""
        if ss.get('chat_with_json_buffer'):
            history = ss.get('chat_with_json_buffer').chat_memory.messages
        summaries_prompt = PromptTemplate(template=ChatPrompt.json_query_prompt_summaries,
                                          input_variables=["query", "chat_history"])
        chain = LLMChain(llm=llm, prompt=summaries_prompt)
        query_summaries = chain.run({
            "query": query,
            "chat_history": history
        })
        return query_summaries

    @staticmethod
    @timer_decorator
    def chat_with_json(query: str, uploaded_file: dict):
        temperature = ss.get("OPENAI_TEMPERATURE") if ss.get("OPENAI_TEMPERATURE") else 0.0
        engine = st.session_state.get("OPENAI_MODEL")
        chat_history = ss.get('chat_with_json_buffer') if ss.get(
            'chat_with_json_buffer') else ConversationBufferMemory(memory_key="chat_history")
        llm = get_llm(model="GPT-4", temperature=temperature)
        # query_summaries = ChatService.get_summer(query=query,
        #                                          llm=llm)
        st.chat_message("user").write(f'Generating for: **{query}**')
        uploaded_file = JsonAgents.extract_keys(data=uploaded_file, keys_to_extract=keys_to_extract)
        json_agent = JsonAgents(
            engine=engine,
            human_query=query,
            uploaded_data=uploaded_file,
            temperature=temperature
        )
        num_tokens = CountTokens.num_tokens_from_text(text=json.dumps(uploaded_file),
                                                      model=model_dict[engine]['model_name'])
        if engine == "GPT-4" and num_tokens > 8000:
            st.warning(
                f"""The total amount of tokens in the current file after extracting keywords is {num_tokens}, 
                please use the GPT-3-16K or GPT-4-32K""",
                icon="⚠️"
            )
            return
        tools = [
            Tool(
                name="specific",
                func=json_agent.json_chat_overall,
                description="Useful when you need to answer something specific about a component described by JSON.",
                return_direct=True
            ),
            Tool(
                name="identify",
                func=json_agent.identify_gpt,
                description="Useful when you need to identify or recognize the component",
                return_direct=True
            ),
        ]
        agent_prompt = CustomPromptTemplate(
            template=ChatPrompt.agent_template,
            tools=tools,
            input_variables=["input", "intermediate_steps"],
        )
        output_parser = CustomOutputParser()
        llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                            tools=tools,
                                                            verbose=True)
        with get_openai_callback() as cb:
            with st.chat_message("assistant", avatar="🤔"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.run(query, callbacks=[st_callback])
            print(cb)
            # response, cb = json_agent.json_chat_overall(query=query)
            # response, cb = json_agent.json_chat_detail(query=query)

        ss['chat_with_json_cost'] = [get_cost_dict(cb)] + [{}] + ss.get('chat_with_json_cost', [])
        chat_history.save_context({'input': query}, {'output': response})
        ss['chat_with_json_buffer'] = chat_history
        return cb


@dataclass
class JsonAgents:
    engine: str
    human_query: str
    uploaded_data: dict
    temperature: float = 0.0
    llm: AzureChatOpenAI = field(init=False)
    chat_history: ConversationBufferMemory = field(init=False)

    def __post_init__(self):
        self.llm = get_llm(model=self.engine, temperature=self.temperature)
        self.chat_history = ss.get('chat_with_json_buffer') if ss.get(
            'chat_with_json_buffer') else ConversationBufferMemory(memory_key="chat_history")

    def json_chat_detail(self, query: str):
        json_spec = JsonSpec(dict_=self.uploaded_data, max_value_length=2000)
        json_toolkit = JsonToolkit(spec=json_spec)
        agent_executor = create_json_agent(
            llm=self.llm,
            toolkit=json_toolkit,
            verbose=True,
            suffix=ChatPrompt.json_re_suffix,
            input_variables=["input", "agent_scratchpad", "chat_history"]
        )
        with get_openai_callback() as cb:
            with st.chat_message("assistant", avatar="🤔"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.run({"input": self.human_query, "chat_history": self.chat_history.buffer},
                                              callbacks=[st_callback])
        return response, cb

    def json_chat_overall(self, query: str):
        json_overall_prompt = PromptTemplate(template=ChatPrompt.json_overall_prompt,
                                             input_variables=["query", "chat_history", "data"])
        chain = LLMChain(llm=self.llm, prompt=json_overall_prompt, verbose=True)
        response = chain.run({
            "query": self.human_query,
            "chat_history": self.chat_history.buffer,
            "data": self.uploaded_data
        })
        return response

    def identify_combine(self, query: str):
        json_identify_prompt = PromptTemplate(template=ChatPrompt.json_identify_prompt,
                                              input_variables=["query", "data"])
        chain = LLMChain(llm=self.llm, prompt=json_identify_prompt, verbose=True)
        response = chain.run({
            "query": self.human_query,
            "data": self.uploaded_data
        })
        if "Answer:" in response:
            gpt_response = response.split("Answer:")[-1]
        else:
            gpt_response = response
        gpt_response = json.loads(gpt_response)
        milvus_host = os.environ['MILVUS_HOST']
        embeddings = get_embedding("Ada-embedding")
        component_instance = Milvus(
            collection_name="dev_figma_components",
            embedding_function=embeddings,
            connection_args={"host": milvus_host},
        )
        extract_data = json.dumps(self.uploaded_data)
        json_docs = component_instance.similarity_search_with_score(
            query=extract_data,
            k=4
        )
        vector_response = list()
        for doc in json_docs:
            types = doc[0].metadata["type"]
            probability = (1 - (round(doc[1], 2) / 0.5))
            vector_response.append(
                {
                    "Type": types,
                    "Probability": "{:.1%}".format(probability)
                }
            )
        fin_response = list()
        for i in vector_response:
            similarity = fuzz.token_set_ratio(i["Type"], gpt_response["Type"])
            if similarity > 50:
                fin_response.append({
                    "Type": i["Type"],
                    "Probability": f'{(float(i["Probability"].strip("%")) + float(gpt_response["Probability"].strip("%"))) / 2}%'
                })
            else:
                fin_response.append({
                    "Type": i["Type"],
                    "Probability": f'{float(i["Probability"].strip("%"))}%'
                })
        response = json.dumps(fin_response)
        return response

    def identify_vector(self, query: str):
        milvus_host = os.environ['MILVUS_HOST']
        embeddings = get_embedding("Ada-embedding")
        component_instance = Milvus(
            collection_name="dev_figma_components",
            embedding_function=embeddings,
            connection_args={"host": milvus_host},
        )
        extract_data = json.dumps(self.uploaded_data)
        json_docs = component_instance.similarity_search_with_score(
            query=extract_data,
            k=4
        )
        response = list()
        for doc in json_docs:
            types = doc[0].metadata["type"]
            probability = (1 - (round(doc[1], 2) / 0.5))
            response.append(
                {
                    "Type": types,
                    "Probability": "{:.1%}".format(probability)
                }
            )
        return json.dumps(response)

    def identify_gpt(self, query: str):
        json_identify_prompt = PromptTemplate(template=ChatPrompt.json_identify_prompt,
                                              input_variables=["query", "data"])
        chain = LLMChain(llm=self.llm, prompt=json_identify_prompt, verbose=True)
        response = chain.run({
            "query": self.human_query,
            "data": self.uploaded_data
        })
        if "Answer:" in response:
            response = response.split("Answer:")[-1]

        return response

    @staticmethod
    def extract_keys(data: dict, keys_to_extract: list) -> dict:
        extracted_data = {}
        for key, value in data.items():
            if key in keys_to_extract:
                extracted_data[key] = value
            if key == "children":
                if isinstance(value, dict):
                    extracted_data[key] = JsonAgents.extract_keys(value, keys_to_extract)
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, dict):
                            value[index] = JsonAgents.extract_keys(item, keys_to_extract)
                    extracted_data[key] = value
        return extracted_data
