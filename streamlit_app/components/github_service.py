import asyncio
import os
import time
import json
import pandas as pd
from typing import Tuple
from components.custom_loaders import CustomGitLoader
from concurrent.futures import ThreadPoolExecutor
from components.postgre_wrapper import PgService
from modules.model import get_llm
from datetime import datetime
from dotenv import load_dotenv
from github import Github
from github.Repository import Repository
from github.PaginatedList import PaginatedList
from dataclasses import dataclass, field
from components.key_vault import FetchKey
from components.chat_prompt import ChatPrompt
from modules.model import get_embedding
from modules.mass_milvus import MassMilvus
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate

load_dotenv(verbose=True)


@dataclass
class GithubClient:
    """The class that defines the GitHub client as source"""

    base_url: str = ""
    login_token: str = field(init=False)
    github: Github = field(init=False)

    def __post_init__(self):
        self.login_token = GithubClient.get_key_vault()
        self.github = self.git_conn()

    @staticmethod
    def get_key_vault() -> str:
        if len(os.getenv("GITHUB_TOKEN")) > 0:
            return os.getenv("GITHUB_TOKEN")
        else:
            return FetchKey("GITHUB-TOKEN").retrieve_secret()

    def git_conn(self) -> Github:
        if len(self.login_token):
            if self.base_url:
                return Github(base_url=self.base_url, login_or_token=self.login_token)
            return Github(login_or_token=self.login_token)
        else:
            print("No github token")


@dataclass
class GithubService:
    """The class defines the GitHub service to fetch repo releases, Prs, codes change"""

    github: Github

    @staticmethod
    def get_repo_name(git_url: str) -> str:
        url_list = git_url.split("/")
        print(url_list)
        if "github.com" in url_list or "gitlab.com" in url_list:
            repo_name = f"{url_list[3]}/{url_list[4]}"
            return repo_name
        return ""

    def get_repo(self, name: str) -> Repository:
        repository = self.github.get_repo(name)
        return repository

    @staticmethod
    def get_releases(repo: Repository) -> list:
        releases_list = list()
        releases = repo.get_releases()
        if releases:
            try:
                releases_list = [releases[0], releases[1]]
            except IndexError:
                releases_list = [releases[0]]
        return releases_list

    @staticmethod
    def get_repo_pulls(repo: Repository, end_time: datetime) -> list:
        pull_list = list()
        pulls = repo.get_pulls(state='closed', sort='created', direction='desc')
        for pull in pulls:
            if pull.merged_at and end_time:
                if pull.merged_at >= end_time:
                    pull_list.append({
                        "merged_at": pull.merged_at,
                        "pull": pull,
                        "file_changes": pull.get_files()
                    })
                else:
                    break
            else:
                if pull.merged_at:
                    pull_list.append({
                        "merged_at": pull.merged_at,
                        "pull": pull,
                        "file_changes": pull.get_files()
                    })
        return pull_list

    @staticmethod
    def get_issues(repo: Repository) -> PaginatedList:
        open_issues = repo.get_issues(state="open", labels=["bug"])
        try:
            open_issues[0]
        except IndexError:
            open_issues = repo.get_issues(state="open")
        return open_issues

    @staticmethod
    def release_notes_content(repo_name: str, pulls: list, table_name: str, releases: list = None,
                              issues: PaginatedList = None) -> list:
        pg_service = PgService(table_name=table_name)
        pull_files_changed = list()
        issues_content = list()
        for pull in pulls:
            for file in pull["file_changes"]:
                file_contents = file.patch
                if file_contents and "image/png" in file_contents:
                    file_contents = ""
                changed = {
                    "file_name": file.filename,
                    "changed_content": file_contents
                }
                pull_files_changed.append(changed)
        if issues:
            for issue in issues:
                issues_content.append(issue.title)
        else:
            issues_content.append("Did not get issue information")
        json_pulls = json.dumps(pull_files_changed, ensure_ascii=False)
        json_pulls = GithubService.summary_pull_content(json_pulls)
        json_issues = json.dumps(issues_content, ensure_ascii=False)
        contents_dict = {
            "repo_name": repo_name,
            "release_version": releases[0].tag_name if releases else "v0.0.1",
            "release_date": releases[0].published_at.strftime("%Y-%m-%d") if releases else datetime.now().strftime(
                "%Y-%m-%d"),
            "release_body": releases[0].body if releases else "",
            "release_change_contents": json_pulls,
            "issues_contents": json_issues
        }
        df = pd.DataFrame(data=[contents_dict])
        pg_service.handle_table(data_df=df)
        return [contents_dict]

    @staticmethod
    def store_release_notes(repo_name: str, release_version: str, release_note: json, table_name: str,):
        pg_service = PgService(table_name=table_name)
        release_dict = {
            "repo_name": repo_name,
            "release_version": release_version,
            "release_note": release_note
        }
        df = pd.DataFrame(data=[release_dict])
        pg_service.handle_table(data_df=df)

    @staticmethod
    def summary_pull_content(contents: str) -> str:
        llm = get_llm(model="GPT-4", temperature=0.2)
        pull_changes = contents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000,
                                                       chunk_overlap=100)
        texts = text_splitter.split_text(pull_changes)
        docs = [Document(page_content=t) for t in texts]
        summaries_prompt = PromptTemplate(template=ChatPrompt.pr_summaries,
                                          input_variables=["text"])
        print(len(docs))
        summaries_chain = load_summarize_chain(llm=llm,
                                               chain_type="map_reduce",
                                               map_prompt=summaries_prompt,
                                               combine_prompt=summaries_prompt)
        summ_changes = summaries_chain.run(docs)
        print(summ_changes)
        return summ_changes


@dataclass
class GitRepoLoader:
    """The class defines the GitHub repo loader used by langchain"""

    git_url: str
    git_branch: str = "master"
    file_filter: Tuple[str, ...] = None
    local_path: str = field(init=False)

    def __post_init__(self):
        split_git_url = self.git_url.split("/")
        self.local_path = f"./{split_git_url[3]}/{split_git_url[4]}"

    def git_load_remote(self):
        loader = CustomGitLoader(
            clone_url=self.git_url,
            repo_path=self.local_path,
            branch=self.git_branch,
        )

        # data_doc = loader.load() #for loop function
        data_doc = asyncio.run(loader.a_load())
        data_doc = [doc for doc in data_doc if doc.page_content]
        return data_doc

    def git_load_local(self):
        if self.file_filter:
            loader = CustomGitLoader(
                repo_path=self.local_path,
                branch=self.git_branch,
                file_filter=lambda file_path: file_path.endswith(self.file_filter)
            )
        else:
            loader = CustomGitLoader(
                repo_path=self.local_path,
                branch=self.git_branch,
            )
        # data_doc = loader.load() #for loop function
        data_doc = asyncio.run(loader.a_load())
        data_doc = [doc for doc in data_doc if doc.page_content]
        return data_doc

    def get_releases_version(self):
        release_version = ""
        git_conn = GithubClient().git_conn()
        svc = GithubService(github=git_conn)
        repo_name = svc.get_repo_name(self.git_url)
        if repo_name:
            releases = svc.get_releases(repo=svc.get_repo(repo_name))
            if releases:
                release_version = releases[0].tag_name
                release_version = release_version.replace(".", "_")
        return repo_name, release_version


@dataclass
class GitStoreVector:
    """The class defines that store GitHub repo to vector database"""

    git_url: str
    database_host: str
    database_port: str = "19530"
    git_branch: str = "master"
    model: str = "Ada-embedding"
    chunk_size: int = 300
    chunk_overlap: int = 20
    test_split: RecursiveCharacterTextSplitter = field(init=False)

    def __post_init__(self):
        self.test_split = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_docs(self, docs: list[Document]) -> list[Document]:
        doc_chunks = []

        for doc in docs:
            chunks = self.test_split.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata_copy = doc.metadata.copy()
                metadata_copy['chunk'] = i
                doc = Document(
                    page_content=chunk, metadata=metadata_copy
                )
                doc_chunks.append(doc)

        return doc_chunks

    async def astore_milvus(self, remote: bool = False):
        milvus_connection_args = {'host': self.database_host}
        timestamp = int(time.time() * 1000.0)
        grl = GitRepoLoader(git_url=self.git_url, git_branch=self.git_branch)
        if remote:
            docs = grl.git_load_remote()
        else:
            docs = grl.git_load_local()
        doc_chunks = self.split_docs(docs)
        repo_name, release_version = grl.get_releases_version()
        repo_name = repo_name.split("/")
        if not release_version:
            collection_name = f"dev_git_{repo_name[0]}_{repo_name[1]}_{timestamp}"
        else:
            collection_name = f"dev_git_{repo_name[0]}_{repo_name[1]}_{release_version}"
        collection_name = collection_name.replace("-", "_")
        print(collection_name)
        print(len(doc_chunks))
        futures = [
            asyncio.create_task(AsyncService([doc], collection_name, milvus_connection_args).async_store())
            for doc in doc_chunks
        ]
        status_info = await asyncio.gather(*futures)
        return status_info

    def store_milvus(self, remote: bool = False):
        milvus_connection_args = {'host': self.database_host}
        timestamp = int(time.time() * 1000.0)
        grl = GitRepoLoader(git_url=self.git_url, git_branch=self.git_branch)
        if remote:
            docs = grl.git_load_remote()
        else:
            docs = grl.git_load_local()
        doc_chunks = self.split_docs(docs)
        repo_name, release_version = grl.get_releases_version()
        repo_name = repo_name.split("/")
        if not release_version:
            collection_name = f"dev_git_{repo_name[0]}_{repo_name[1]}_{timestamp}"
        else:
            collection_name = f"dev_git_{repo_name[0]}_{repo_name[1]}_{release_version}"
        collection_name = collection_name.replace("-", "_")
        print(collection_name)
        print(len(doc_chunks))
        count = 1
        embeddings = get_embedding("Ada-embedding")
        for doc in doc_chunks:
            print(count)
            count += 1
            Milvus.from_documents([doc],
                                  embeddings,
                                  connection_args=milvus_connection_args,
                                  collection_name=collection_name)


@dataclass
class AsyncService:
    """The class that defines th async service"""

    doc: list[Document]
    collection_name: str
    connection_args: dict
    model: str = "Ada-embedding"
    embedding: get_embedding = field(init=False)

    def __post_init__(self):
        self.embedding = get_embedding(self.model)

    async def async_store(self):
        return await asyncio.to_thread(Milvus.from_documents,
                                       documents=self.doc,
                                       embedding=self.embedding,
                                       collection_name=self.collection_name,
                                       connection_args=self.connection_args)

    @staticmethod
    async def a_similarity_search(collection_instance: Milvus, query: str, k: int = 4):
        return await asyncio.to_thread(
            collection_instance.similarity_search_with_score,
            query=query,
            k=k)


if __name__ == "__main__":
    # executor = ThreadPoolExecutor(max_workers=100)
    # asyncio.get_event_loop().set_default_executor(executor)
    os.environ["OPENAI_API_KEY"] = FetchKey("OPENAI-KEY").retrieve_secret()
    MILVUS_HOST = '52.226.226.29'
    milvus_connection_args = {'host': MILVUS_HOST}
    git_url = "https://github.com/theskumar/python-dotenv"
    grl = GitRepoLoader(git_url="https://github.com/hwchase17/langchain",
                        git_branch="master",
                        file_filter=(".py",))
    data = grl.git_load_local()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    docs = text_splitter.split_documents(data)
    print('Total number of documents:', len(docs))
    ts = time.time()
    a = MassMilvus.afrom_documents(docs, collection_name='dev_git_hwchase17_langchain_codes',
                                   connection_args=milvus_connection_args,
                                   drop_old=True)
    print('Total time usage:', f'{time.time() - ts:.2f}')
