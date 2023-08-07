import os
import stat
import shutil
import json
import asyncio
import time
from typing import List
from git import Repo
from langchain.document_loaders import GitLoader
from langchain.docstore.document import Document


class CustomGitLoader(GitLoader):

    def load(self) -> List[Document]:
        start = time.perf_counter()
        try:
            from git import Blob, Repo  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "Could not import git python package. "
                "Please install it with `pip install GitPython`."
            ) from ex

        if not os.path.exists(self.repo_path) and self.clone_url is None:
            raise ValueError(f"Path {self.repo_path} does not exist")
        elif self.clone_url:
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path, onerror=CustomGitLoader.remove_readonly)
            repo = Repo.clone_from(self.clone_url, self.repo_path)
            repo.git.checkout(self.branch)
        else:
            repo = Repo(self.repo_path)
            repo.git.checkout(self.branch)

        docs: List[Document] = []
        print(f"Duration checkout- {round(time.perf_counter() - start, 2)} seconds.")

        for item in repo.tree().traverse():
            if not isinstance(item, Blob):
                continue

            file_path = os.path.join(self.repo_path, item.path)

            ignored_files = repo.ignored([file_path])  # type: ignore
            if len(ignored_files):
                continue

            # uses filter to skip files
            if self.file_filter and not self.file_filter(file_path):
                continue

            rel_file_path = os.path.relpath(file_path, self.repo_path)
            file_name = item.name
            file_type = os.path.splitext(item.name)[1]
            try:
                if file_name.endswith(".ipynb"):
                    text_content = CustomGitLoader.read_notebook_without_images(file_path=file_path)
                else:
                    content = CustomGitLoader.read_file(file_path=file_path)
                    # loads only text files
                    try:
                        text_content = content.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                metadata = {
                    "source": rel_file_path,
                    "file_path": rel_file_path,
                    "file_name": file_name,
                    "file_type": file_type,
                }
                doc = Document(page_content=text_content, metadata=metadata)
                docs.append(doc)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        return docs

    async def a_load(self) -> List[Document]:
        """
        # limit the maximum number of threads
        executor = ThreadPoolExecutor(max_workers=100)
        asyncio.get_event_loop().set_default_executor(executor)
        """
        start = time.perf_counter()
        try:
            from git import Blob, Repo  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "Could not import git python package. "
                "Please install it with `pip install GitPython`."
            ) from ex

        if not os.path.exists(self.repo_path) and self.clone_url is None:
            raise ValueError(f"Path {self.repo_path} does not exist")
        elif self.clone_url:
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path, onerror=CustomGitLoader.remove_readonly)
            repo = Repo.clone_from(self.clone_url, self.repo_path)
            repo.git.checkout(self.branch)
        else:
            repo = Repo(self.repo_path)
            repo.git.checkout(self.branch)
        print(f"Duration checkout- {round(time.perf_counter() - start, 2)} seconds.")
        docs_futures = []
        for item in repo.tree().traverse():
            if not isinstance(item, Blob):
                continue
            file_path = os.path.join(self.repo_path, item.path)

            # uses filter to skip files
            if self.file_filter and not self.file_filter(file_path):
                continue
            docs_futures.append(
                asyncio.create_task(CustomGitLoader.load_thread(
                    repo=repo,
                    file_path=file_path,
                    repo_path=self.repo_path,
                    file_name=item.name,
                ))
            )
        print(f"Duration add-tasks - {round(time.perf_counter() - start, 2)} seconds.")
        return await asyncio.gather(*docs_futures)

    @staticmethod
    def process_each_file(repo: Repo, file_path: str, repo_path: str, file_name: str) -> Document:
        rel_file_path = os.path.relpath(file_path, repo_path)
        metadata = {
            "source": rel_file_path,
            "file_path": rel_file_path,
            "file_name": file_name,
            "file_type": os.path.splitext(file_name)[1],
        }
        doc = Document(page_content="", metadata=metadata)
        ignored_files = repo.ignored([file_path])  # type: ignore
        if len(ignored_files):
            return doc
        if file_name.endswith(".ipynb"):
            try:
                text_content = CustomGitLoader.read_notebook_without_images(file_path=file_path)
                doc = Document(page_content=text_content, metadata=metadata)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        else:
            try:
                content = CustomGitLoader.read_file(file_path=file_path)
                # loads only text files
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    return doc
                doc = Document(page_content=text_content, metadata=metadata)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        return doc

    @staticmethod
    async def load_thread(repo: Repo, file_path: str, repo_path: str, file_name: str):
        return await asyncio.to_thread(
            CustomGitLoader.process_each_file,
            repo=repo,
            file_path=file_path,
            repo_path=repo_path,
            file_name=file_name
        )

    @staticmethod
    def read_notebook_without_images(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            notebook = json.load(file)
        filtered_cells = [cell['source'] for cell in notebook['cells']]
        return '\n'.join(['\n'.join(cell) for cell in filtered_cells])

    @staticmethod
    def read_file(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
        return content

    @staticmethod
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)
