import asyncio
import json
import os
import pandas as pd
import streamlit as st
from typing import List, Tuple
from dataclasses import dataclass, field
from components.collect_answers import CollectAnswers
from components.github_service import GithubService, GithubClient, GitStoreVector
from components.html_templates import user_template, bot_template
from components.chat_prompt import ChatPrompt
from modules.model import get_embedding, get_llm
from components.chat_prompt import CustomPromptTemplate, CustomOutputParser
from components.postgre_wrapper import PgService
from components.frontend_cost import get_cost_dict, timer_decorator
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chat_models import AzureChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks import get_openai_callback
from langchain import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from components.github_service import AsyncService
from pymilvus import connections, utility

ss = st.session_state


def clear_submit():
    st.session_state["submit"] = False


def initialize_session_state():
    ss.pop('chat_with_git_buffer', None)
    ss.pop('chat_with_git_cost', None)
    ss.pop('release_contents', None)
    ss.pop('has_release_notes', None)
    ss.pop('git_sources', None)
    ss.pop('time_spent_chat_with_git', None)
    ss.pop('collect_search', None)
    ss.pop('git_answers_share', None)
    ss.pop('release_version', None)
    ss.pop('git_near_answers', None)


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
        submit_col, clear_col, _, _, _, regenerate_col = st.columns([2, 2, 3, 2, 3, 3])
        with submit_col:
            button_submit = st.button('Submit', type="primary")
        with clear_col:
            button_clear = st.button('Clear')
        with regenerate_col:
            button_regenerate = st.button("Regenerate")

        if button_clear:
            initialize_session_state()
            ss["submit"] = False

        if button_submit or st.session_state.get("submit"):
            ss["submit"] = True
            if question:
                with st.spinner(":green[Generating...]"):
                    ChatService.chat_with_git(question)

        if button_regenerate:
            if question:
                with st.spinner(":green[Generating...]"):
                    ChatService.chat_with_git(question, regenerate=True)

    @staticmethod
    def get_chat_record():
        messages = ss.get('chat_with_git_buffer', ConversationBufferMemory()).chat_memory.messages.copy()
        clt = CollectAnswers(pg_table_name="likes_answers_collection")
        messages.reverse()
        if ss.get("release_contents"):
            RenderPage.process_text_contents()
        if ss.get('git_sources'):
            with st.expander(':sparkles: Sources'):
                source = ss['git_sources']
                st.data_editor(pd.DataFrame(source),
                               use_container_width=True,
                               hide_index=True,
                               disabled=True,)
        if ss.get("git_near_answers"):
            with st.expander(':bulb: Similar Answers'):
                for each in ss.get("git_near_answers"):
                    # with st.expander(f":question: {each[0]}"):
                    st.chat_message("user").write(each[1])
                    st.chat_message("assistant").write(each[2])
        for i, msg in enumerate(messages):
            if i % 2 == 0:
                st.chat_message("assistant").write(
                    bot_template.format(MSG=msg.content,
                                        MODEL=ss['chat_with_git_cost'][i]['model'],
                                        COST=round(ss['chat_with_git_cost'][i]['cost'], 6),
                                        TOKENS_USED=ss['chat_with_git_cost'][i]['total_tokens'],
                                        PROMPT=ss['chat_with_git_cost'][i]['prompt'],
                                        COMPLETION=ss['chat_with_git_cost'][i]['completion'],
                                        TIME=ss['time_spent_chat_with_git'][i]),
                    unsafe_allow_html=True)
                if ss.get("collect_search")[i]:
                    thumbs_num = int(ss.get("collect_search")[i][4])
                else:
                    thumbs_num = 0
                clt.collect_answers(collect_search=st.session_state["collect_search"][i],
                                    index=i,
                                    user_name=ss["USER_INFO"]["username"],
                                    human_query=messages[i + 1].content,
                                    bot_answers=msg.content,
                                    )
                col1, col2, col3, _, _, col4 = st.columns([5, 1, 1, 3, 3, 3])
                with col1:
                    st.markdown("###### Was this response helpful?")
                with col2:
                    thumbs_up = st.button("👍", help=f"Likes: {thumbs_num}", key=f"thumbs_up_{i}")
                with col3:
                    thumbs_down = st.button("👎", key=f"thumbs_down_{i}")
                if thumbs_up:
                    clt.collect_answers(thumbs="up",
                                        collect_search=st.session_state["collect_search"][i],
                                        index=i,
                                        user_name=ss["USER_INFO"]["username"],
                                        human_query=messages[i + 1].content,
                                        bot_answers=msg.content,
                                        )
                    st.success("Thank you for your feedback!", icon="✅")
                if thumbs_down:
                    clt.collect_answers(thumbs="down",
                                        collect_search=st.session_state["collect_search"][i],
                                        index=i,
                                        user_name=ss["USER_INFO"]["username"],
                                        human_query=messages[i + 1].content,
                                        bot_answers=msg.content,
                                        )
                    st.success("Thank you for your feedback!", icon="✅")
            else:
                st.chat_message("user").write(msg.content)
                st.write("---")

    @staticmethod
    def process_text_contents():
        st.title("Release Notes")
        release_notes = ss.get("release_contents")
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

    @staticmethod
    @st.cache_data(ttl=120)
    def display_conversation_history():
        history_info = CollectAnswers.collect_history_conversation(table_name="likes_answers_collection",
                                                                   user_name=ss["USER_INFO"]["username"])
        if history_info:
            st.subheader(':point_right: Conversation History')
            for each in history_info:
                with st.expander(f":question: {each[0]}"):
                    st.chat_message("assistant").write(each[1])


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
    @st.cache_resource(ttl=600)
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
            ss["release_version"] = releases[0].tag_name
            pg_service = PgService(table_name=table_name)
            check_sql = f"""
                       SELECT repo_name, release_version, release_date, release_body, release_change_contents, issues_contents
                        FROM repo_release_contents WHERE release_version = '{releases[0].tag_name}'
                       and repo_name = '{repo_name}';
                   """
            has_data = pg_service.execute_sql(command=check_sql)
            if has_data:
                check_sql = f"""
                               SELECT release_note
                               FROM repo_release_notes WHERE release_version = '{releases[0].tag_name}'
                               and repo_name = '{repo_name}';
                               """
                has_notes = pg_service.execute_sql(command=check_sql)
                if has_notes:
                    ss["has_release_notes"] = True
                    return has_notes[0][0]
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
    @timer_decorator
    def chat_with_git(query: str, regenerate: bool = False):
        ss.pop('release_contents', None)
        ss.pop('has_release_notes', None)
        ss.pop('git_near_answers', None)
        ss.pop('git_sources', None)
        temperature = ss.get("OPENAI_TEMPERATURE") if ss.get("OPENAI_TEMPERATURE") else 0.0
        engine = st.session_state.get("OPENAI_MODEL")
        chat_history = ss.get('chat_with_git_buffer') if ss.get(
            'chat_with_git_buffer') else ConversationBufferMemory()
        query_summaries = ChatService.get_summer(query=query,
                                                 temperature=temperature)
        st.chat_message("user").write(f'Searching for: **{query_summaries}**')

        if not regenerate:
            """Find out if similar questions and answers are included"""
            git_likes_instance = ChatService.get_milvus_instance(collection_name="likes_answers_collection")
            collect_docs = asyncio.run(GithubAgent.get_docs(query=query,
                                                            git_instance=git_likes_instance,
                                                            k=3))
            CollectAnswers.collect_search(
                docs=collect_docs,
                table_name="likes_answers_collection"
            )

        agent = GithubAgent(query_summaries=query_summaries,
                            temperature=temperature,
                            engine=engine,
                            human_query=query,
                            regenerate=regenerate)
        tools = [
            Tool(
                name="Common chat",
                func=agent.repo_chat_classification,
                description="useful for when you need to answer questions except release notes/release and version",
                return_direct=True
            ),
            Tool(
                name="Release note generator",
                func=agent.repo_release_notes,
                description="useful only for when you need to generate or provide a release notes",
                return_direct=True
            ),
            Tool(
                name="Specific chat about the releases",
                func=agent.repo_release_chat_toolkit,
                description="Useful when you need to answer questions about release, except release notes",
                return_direct=True
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
            with st.chat_message("assistant", avatar="🤔"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.run(query_summaries, callbacks=[st_callback])
            print(cb)
        chat_history.save_context({'input': query}, {'output': response})
        ss['chat_with_git_buffer'] = chat_history
        collection_list = ss.get("selected_git_collection").split("_")
        if "hwchase17" not in collection_list:
            repo_name = f"{collection_list[2]}/{'-'.join(collection_list[3:])}"
            if "-v" in repo_name:
                repo_name = repo_name.split("-v")[0]
        else:
            repo_name = f"{collection_list[2]}/{collection_list[3]}"
        if not ss.get("has_release_notes") and ss.get("release_contents"):
            GithubService.store_release_notes(
                repo_name=repo_name,
                release_version=ss.get("release_version"),
                table_name="repo_release_notes",
                release_note=ss.get("release_contents")
            )
        ss['chat_with_git_cost'] = [get_cost_dict(cb)] + [{}] + ss.get('chat_with_git_cost', [])
        return cb


@dataclass
class GithubAgent:
    engine: str
    human_query: str
    query_summaries: str
    # docs: List[List[Tuple[Document, float]]]
    regenerate: bool = False
    temperature: float = 0.0
    llm: AzureChatOpenAI = field(init=False)

    def __post_init__(self):
        self.llm = get_llm(model=self.engine, temperature=self.temperature)

    def repo_release_notes(self, query: str):
        if not ss.get("collect_search")[0] or self.regenerate:
            response = ChatService.get_release_notes()
            if not ss.get("has_release_notes"):
                release_prompt = PromptTemplate(template=ChatPrompt.release_prompt,
                                                input_variables=["text", "question"])
                release_chain = LLMChain(llm=self.llm,
                                         prompt=release_prompt)
                response = release_chain.run({"question": query, "text": response[0]})
            ss["release_contents"] = response
            if self.regenerate:
                st.session_state["collect_search"] = [tuple()] + [tuple()] + st.session_state.get("collect_search",
                                                                                                  [])
        else:
            response = ss.get("collect_search")[0][2]
            ss["has_release_notes"] = True
            ss["release_contents"] = response
        return response

    # def repo_release_chat(self, query: str):
    #     pg_user = PgService.user
    #     pg_password = PgService.password
    #
    #     pg_connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@c.gpt-project-cosmos-db-postgresql.postgres.database.azure.com:5432/citus"
    #
    #     db = SQLDatabase.from_uri(pg_connection_string,
    #                               include_tables=['repo_release_contents'],
    #                               sample_rows_in_table_info=2,
    #                               max_string_length=3000
    #                               )
    #
    #     db_chain = SQLDatabaseChain.from_llm(llm=self.llm,
    #                                          db=db,
    #                                          top_k=10,
    #                                          verbose=True)
    #     response = db_chain.run(self.human_query)
    #     return response

    def repo_release_chat_toolkit(self, query: str):
        if not ss.get("collect_search")[0] or self.regenerate:
            pg_user = PgService.user
            pg_password = PgService.password

            pg_connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@c.gpt-project-cosmos-db-postgresql.postgres.database.azure.com:5432/citus"

            db = SQLDatabase.from_uri(pg_connection_string,
                                      include_tables=['repo_release_notes'],
                                      sample_rows_in_table_info=2,
                                      max_string_length=3000
                                      )
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            agent_executor = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.run(self.human_query,
                                          callbacks=[st_callback])
            if self.regenerate:
                st.session_state["collect_search"] = [tuple()] + [tuple()] + st.session_state.get("collect_search",
                                                                                                  [])
        else:
            response = ss.get("collect_search")[0][2]
        return response

    def repo_chat_classification(self, query: str):
        if not ss.get("collect_search")[0] or self.regenerate:
            chat_history = ss.get('chat_with_git_buffer') if ss.get(
                'chat_with_git_buffer') else ConversationBufferMemory()
            git_instance = ss.get("milvus_git_instance")
            docs = asyncio.run(GithubAgent.get_docs(query=self.query_summaries,
                                                    git_instance=git_instance,
                                                    k=6))
            modified_docs = docs.copy()
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
                {'input_documents': modified_docs, 'question': self.human_query, 'chat_history': chat_history.buffer})
            collection_list = ss.get("selected_git_collection").split("_")
            if "hwchase17" not in collection_list:
                repo_name = f"{collection_list[2]}/{'-'.join(collection_list[3:])}"
                if "-v" in repo_name:
                    repo_name = repo_name.split("-v")[0]
                    repo_link = f"https://github.com/{repo_name}/tree/main/"
                else:
                    repo_link = f"https://gitlab.com/{repo_name}/-/blob/main/"
            else:
                repo_name = f"{collection_list[2]}/{collection_list[3]}"
                repo_link = f"https://github.com/{repo_name}/tree/master/"
            sources = [{
                'score': round(d[1], 5),
                'content': d[0].page_content,
                'file_path': f"{repo_link}{d[0].metadata['source']}",
                'file_name': d[0].metadata["file_name"],
            } for doc in docs for d in doc]
            ss['git_sources'] = sources
            response = response['output_text']
            if self.regenerate:
                st.session_state["collect_search"] = [tuple()] + [tuple()] + st.session_state.get("collect_search",
                                                                                                  [])
        else:
            response = ss.get("collect_search")[0][2]

        return response

    @staticmethod
    async def get_docs(query: str, git_instance: list, k: int = 4):
        tasks = [
            asyncio.create_task(AsyncService.a_similarity_search(collection_instance=instance, query=query, k=k))
            for instance in git_instance]

        docs = await asyncio.gather(*tasks)
        return docs
