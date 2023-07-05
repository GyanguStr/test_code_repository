import asyncio
import json
import os
import pandas as pd
import streamlit as st
from typing import List, Tuple
from dataclasses import dataclass, field
from components.github_service import GithubService, GithubClient, GitStoreVector
from components.html_templates import user_template, bot_template, css
from components.chat_prompt import ChatPrompt
from modules.model import get_embedding, get_llm
from components.chat_prompt import CustomPromptTemplate, CustomOutputParser
from components.postgre_wrapper import PgService
from components.frontend_cost import get_cost_dict
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks import get_openai_callback
from langchain import SQLDatabase, SQLDatabaseChain
from components.github_service import AsyncService
from pymilvus import connections, utility

ss = st.session_state


def clear_submit():
    st.session_state["submit"] = False


def initialize_session_state():
    ss.pop('chat_with_git_buffer', None)
    ss.pop('chat_with_git_cost', None)
    ss.pop('release_notes', None)
    ss.pop('git_sources', None)


class RenderPage:
    """This class defines the functions required by the page"""

    @staticmethod
    def get_collections(database_host: str, database_port: str = "19530") -> list:
        if not st.session_state.get("collection_git_list"):
            connections.connect(host=database_host, port=database_port)
            collections = utility.list_collections()
            new_collections = list()
            for item in collections:
                if "git" in item:
                    if 'hwchase17_langchain' not in item:
                        new_collections.append(item)
                    else:
                        if "dev_git_hwchase17_langchain" not in new_collections:
                            new_collections.append("dev_git_hwchase17_langchain")
            new_collections.sort()
            st.session_state["collection_git_list"] = new_collections
        return st.session_state.get("collection_git_list")

    @staticmethod
    def get_question():
        question = st.text_area('question', key='question', height=100, placeholder='Enter question here', help='',
                                label_visibility="collapsed", on_change=clear_submit())
        submit_col, clear_col, _, _, _ = st.columns([1, 1, 2, 2, 2])
        with submit_col:
            button_submit = st.button('Submit', type="primary")
        with clear_col:
            button_clear = st.button('Clear')

        if button_clear:
            initialize_session_state()
            ss["submit"] = False

        if button_submit or st.session_state.get("submit"):
            ss["submit"] = True
            if question:
                ChatService.chat_with_git(question)

    @staticmethod
    def get_chat_record():
        messages = ss.get('chat_with_git_buffer', ConversationBufferMemory()).chat_memory.messages.copy()
        messages.reverse()
        if ss.get("release_notes"):
            RenderPage.process_text_contents()
        for i, msg in enumerate(messages):
            if i % 2 == 0:
                st.write(bot_template.format(MSG=msg.content,
                                             MODEL=ss['chat_with_git_cost'][i]['model'],
                                             COST=round(ss['chat_with_git_cost'][i]['cost'], 6),
                                             TOKENS_USED=ss['chat_with_git_cost'][i]['total_tokens'],
                                             PROMPT=ss['chat_with_git_cost'][i]['prompt'],
                                             COMPLETION=ss['chat_with_git_cost'][i]['completion']),
                         unsafe_allow_html=True)
                with st.expander('Sources'):
                    source = ss['git_sources'][i]
                    st.data_editor(pd.DataFrame(source),
                                   use_container_width=True,
                                   disabled=['score', 'content', 'file_path', 'file_name'],
                                   key=f'weblink_{i}')
            else:
                st.write(user_template.format(MSG=msg.content), unsafe_allow_html=True)

    @staticmethod
    def process_text_contents():
        st.title("Release Notes")
        release_notes = ss.get("release_notes")
        release_notes = json.loads(release_notes)
        if release_notes.get("Release version"):
            st.markdown(f"##### release version---{release_notes['Release version']}")
        for key, value in release_notes.items():
            if key.lower() == "overview":
                st.markdown(f"## {key}")
                st.write(f"- {value}")
            if key.lower() == "release version" or key.lower() == "repository name":
                continue
            if isinstance(value, list):
                if not value:
                    continue
                st.markdown(f"## {key}")
                items = [f"- {each}" for each in value]
                items = '\n'.join(items)
                st.write(items)
                if key.lower() == "known issues":
                    issues_link = f"https://github.com/{release_notes['Repository name']}/issues"
                    st.markdown(
                        f"For detailed issues information, please click to view: *[Issue Details]({issues_link})*")
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
    def get_milvus_instance(collection_name: str, model: str = "Ada-embedding") -> List[Milvus]:
        host = os.getenv("MILVUS_HOST")
        embeddings = get_embedding(model)
        instance_list = list()
        if collection_name == "dev_git_hwchase17_langchain":
            for collcetion in [f"{collection_name}_codes", f"{collection_name}_documents"]:
                git_instance = Milvus(
                    collection_name=collcetion,
                    embedding_function=embeddings,
                    connection_args={"host": host},
                )
                instance_list.append(git_instance)
        else:
            git_instance = Milvus(
                collection_name=collection_name,
                embedding_function=embeddings,
                connection_args={"host": host},
            )
            instance_list.append(git_instance)
        return instance_list

    @staticmethod
    def get_summer(query: str, temperature: float):
        llm = get_llm(model="GPT-4", temperature=temperature)
        history = ""
        if ss.get('chat_with_git_buffer'):
            history = ss.get('chat_with_git_buffer').chat_memory.messages
        summaries_prompt = PromptTemplate(template=ChatPrompt.query_prompt_summaries,
                                          input_variables=["query", "history"])
        chain = LLMChain(llm=llm, prompt=summaries_prompt)
        query_summaries = chain.run({
            "query": query,
            "history": history
        })
        return query_summaries

    @staticmethod
    def get_release_notes(table_name: str = "repo_release_contents", repo_name: str = "") -> list:
        """table_name: Check whether the release contents already exist in the pg table"""

        if not repo_name:
            collection_list = ss.get("selected_git_collection").split("_")
            if "hwchase17" not in collection_list:
                repo_name = f"{collection_list[2]}/{'-'.join(collection_list[3:])}"
                if "-v" in repo_name:
                    repo_name = repo_name.split("-v")[0]
            else:
                repo_name = f"{collection_list[2]}/{collection_list[3]}"
        print(repo_name)
        git_conn = GithubClient().git_conn()
        svc = GithubService(github=git_conn)
        repository = svc.get_repo(repo_name)
        releases = GithubService.get_releases(repository)
        if len(releases) > 1:
            end_time = releases[1].published_at
        else:
            end_time = ""
        if releases and table_name:
            pg_service = PgService(table_name=table_name)
            check_sql = f"""
                       SELECT repo_name, release_version, release_date, release_body, release_change_contents, issues_contents
                        FROM repo_release_contents WHERE release_version = '{releases[0].tag_name}' 
                       and repo_name = '{repo_name}';
                   """
            has_data = pg_service.execute_sql(command=check_sql)
            if has_data:
                columns = ["repo_name", "release_version", "release_date", "release_body", "release_change_contents",
                           "issues_contents"]
                df = pd.DataFrame(data=has_data, columns=columns)
                rst = df.to_dict(orient="records")
                return rst

        pulls = GithubService.get_repo_pulls(repository, end_time)
        issues = GithubService.get_issues(repository)
        rst = GithubService.release_notes_content(
            repo_name=repo_name,
            releases=releases,
            pulls=pulls,
            issues=issues,
            table_name=table_name)
        return rst

    @staticmethod
    def chat_with_git(query: str):
        ss.pop('release_notes', None)
        temperature = ss.get("OPENAI_TEMPERATURE") if ss.get("OPENAI_TEMPERATURE") else 0.0
        engine = st.session_state.get("OPENAI_MODEL")
        chat_history = ss.get('chat_with_git_buffer') if ss.get(
            'chat_with_git_buffer') else ConversationBufferMemory()
        query_summaries = ChatService.get_summer(query=query,
                                                 temperature=temperature)
        st.write(f'Seaching for: **{query_summaries}**')
        git_instance = ss.get("milvus_git_instance")
        docs = asyncio.run(GithubAgent.get_docs(query=query_summaries,
                                                git_instance=git_instance,
                                                k=6))
        agent = GithubAgent(docs=docs,
                            temperature=temperature,
                            engine=engine)
        tools = [
            Tool(
                name="Common chat about the repository",
                func=agent.repo_chat,
                description="useful for when you need to answer questions except release notes/release and version of the repository",
                return_direct=True
            ),
            Tool(
                name="Release note generator",
                func=agent.repo_release_notes,
                description="useful only for when you need to generate or provide a release note",
                return_direct=True
            ),
            Tool(
                name="Specific chat about the release notes",
                func=agent.repo_release_chat,
                description="Useful when you need to answer all release-related questions, except release notes"
            ),
        ]
        agent_prompt = CustomPromptTemplate(
            template=ChatPrompt.agent_template,
            tools=tools,
            input_variables=["input", "intermediate_steps"],
        )
        output_parser = CustomOutputParser()
        llm = get_llm(model=engine, temperature=temperature)
        llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
            memory=chat_history
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        with get_openai_callback() as cb:
            response = agent_executor.run(query)
            print(cb)
        chat_history.save_context({'input': query}, {'output': response})
        ss['chat_with_git_buffer'] = chat_history
        collection_list = ss.get("selected_git_collection").split("_")
        if "hwchase17" not in collection_list:
            repo_name = f"{collection_list[2]}/{'-'.join(collection_list[3:])}"
            if "-v" in repo_name:
                repo_name = repo_name.split("-v")[0]
                repo_link = f"https://github.com/{repo_name}/tree/main/"
            else:
                repo_link = f"https://gitlab.com/{repo_name}/-/blob/main/"
        else:
            repo_link = f"https://github.com/{collection_list[2]}/{collection_list[3]}/tree/master/"
        sources = [{
            'score': round(d[1], 5),
            'content': d[0].page_content,
            'file_path': f"{repo_link}{d[0].metadata['source']}",
            'file_name': d[0].metadata["file_name"],
        } for doc in docs for d in doc]

        ss['git_sources'] = [sources] + [[]] + ss.get('git_sources', [])
        ss['chat_with_git_cost'] = [get_cost_dict(cb)] + [{}] + ss.get('chat_with_git_cost', [])
        return cb


@dataclass
class GithubAgent:
    docs: List[List[Tuple[Document, float]]]
    engine: str
    temperature: float = 0.0
    llm: AzureChatOpenAI = field(init=False)

    def __post_init__(self):
        self.llm = get_llm(model=self.engine, temperature=self.temperature)

    def repo_release_notes(self, query: str):
        contents = ChatService.get_release_notes()
        release_prompt = PromptTemplate(template=ChatPrompt.release_prompt,
                                        input_variables=["text", "question"])
        release_chain = LLMChain(llm=self.llm,
                                 prompt=release_prompt)
        response = release_chain.run({"question": query, "text": contents[0]})
        ss["release_notes"] = response
        return response

    def repo_release_chat(self, query: str):
        prompt = PromptTemplate(
            input_variables=["input", "table_info", "dialect"],
            template=ChatPrompt.db_release_prompt
        )
        pg_user = PgService.user
        pg_password = PgService.password

        pg_connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@c.gpt-project-cosmos-db-postgresql.postgres.database.azure.com:5432/citus"

        db = SQLDatabase.from_uri(pg_connection_string,
                                  include_tables=['repo_release_contents'])

        db_chain = SQLDatabaseChain.from_llm(llm=self.llm,
                                             db=db,
                                             # prompt=prompt,
                                             verbose=True)
        response = db_chain.run(query)
        return response

    def repo_chat(self, query: str):
        chat_history = ss.get('chat_with_git_buffer') if ss.get(
            'chat_with_git_buffer') else ConversationBufferMemory()
        modified_docs = self.docs.copy()
        modified_docs = [
            Document(
                page_content=d[0].page_content,
                metadata=d[0].metadata
            )
            for doc in modified_docs for d in doc
        ]
        prompt = PromptTemplate(template=ChatPrompt.prompt_template,
                                input_variables=['question', 'summaries', 'chat_history'])
        qa_chain = load_qa_with_sources_chain(
            llm=self.llm,
            chain_type='stuff',
            prompt=prompt,
            verbose=True,
        )
        response = qa_chain(
            {'input_documents': modified_docs, 'question': query, 'chat_history': chat_history.buffer})

        return response['output_text']

    @staticmethod
    def repo_chat_classification(query: str, git_instance=ss.get("milvus_git_instance")):
        temperature = ss.get("OPENAI_TEMPERATURE") if ss.get("OPENAI_TEMPERATURE") else 0.0
        engine = st.session_state.get("OPENAI_MODEL")
        chat_history = ss.get('chat_with_git_buffer') if ss.get(
            'chat_with_git_buffer') else ConversationBufferMemory()
        docs = asyncio.run(GithubAgent.get_docs(query=query,
                                                git_instance=git_instance))
        modified_docs = docs.copy()
        modified_docs = [
            Document(
                page_content=d[0].page_content,
                metadata=d[0].metadata
            )
            for doc in modified_docs for d in doc
        ]
        llm = get_llm(model=engine, temperature=temperature)
        prompt = PromptTemplate(template=ChatPrompt.prompt_template,
                                input_variables=['question', 'summaries', 'chat_history'])
        qa_chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type='stuff',
            prompt=prompt,
            verbose=True,
        )
        response = qa_chain(
            {'input_documents': modified_docs, 'question': query, 'chat_history': chat_history.buffer})

        collection_list = ss.get("selected_git_collection").split("_")
        if "hwchase17" not in collection_list:
            repo_name = f"{collection_list[2]}/{'-'.join(collection_list[3:])}"
            if "-v" in repo_name:
                repo_name = repo_name.split("-v")[0]
                repo_link = f"https://github.com/{repo_name}/tree/main/"
            else:
                repo_link = f"https://gitlab.com/{repo_name}/-/blob/main/"
        else:
            repo_link = f"https://github.com/{collection_list[2]}/{collection_list[3]}/tree/master/"
        sources = [{
            'score': round(d[1], 5),
            'content': d[0].page_content,
            'file_path': f"{repo_link}{d[0].metadata['source']}",
            'file_name': d[0].metadata["file_name"],
        } for doc in docs for d in doc]

        ss['git_sources'] = [sources] + [[]] + ss.get('git_sources', [])
        return response['output_text']

    @staticmethod
    async def get_docs(query: str, git_instance: list, k: int = 4):
        tasks = [
            asyncio.create_task(AsyncService.a_similarity_search(collection_instance=instance, query=query, k=k))
            for instance in git_instance]

        docs = await asyncio.gather(*tasks)
        return docs
