from components.postgre_wrapper import PgService
import json
import asyncio
import os
import pandas as pd
from components.github_service import GithubClient, GithubService
from components.collect_answers import CollectAnswers
from modules.chat_with_git import GithubAgent, ChatService
from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec

# table_name = "repo_release_notes"
# pg_service = PgService(table_name=table_name)
#
# release_note = {
#     "Overview": "The iOS & iPadOS 17 SDK provides support to develop apps for iPhone and iPad running iOS & iPadOS 17 beta 3. The SDK comes bundled with Xcode 15 beta 3, available from the Beta Software Downloads. For information on the compatibility requirements for Xcode 15 beta 3, see Xcode 15 Beta 3 Release Notes.",
#     "Release version": "v17.0",
#     "Repository name": "ios",
#     "What's Changed": [
#         "Improved app stability: iOS 17.0 addresses various app crashing issues, providing a smoother and more reliable user experience across all applications.",
#         "Fixed Bluetooth pairing issues: iOS 17.0 resolves issues with Bluetooth pairing, allowing for seamless connections with headphones, speakers, and other Bluetooth-enabled devices.",
#         "Improved GPS accuracy: The update enhances GPS accuracy and location tracking, providing more precise navigation and location-based services.",
#         "Resolved notification glitches: iOS 17.0 fixes issues with delayed or missing notifications, ensuring that users receive timely alerts for messages, emails, and app updates.",
# 		"Fixed storage management issues: The update addresses problems with incorrect storage usage reporting, making it easier for users to manage their device's storage space effectively."
#     ],
#     "New Features": [
#         "Advanced Face ID recognition: Building on the improvements in iOS v16.6, iOS v17.0 offers even faster and more secure facial recognition, with added support for masks and other facial coverings.",
# 		"Customizable Dark Mode: In addition to scheduling, users can now customize the appearance of Dark Mode, adjusting the color scheme and contrast to their personal preferences.",
# 		"Expanded Emoji library: iOS v17.0 introduces even more emojis, with a focus on diverse representation and new animated options for a more expressive messaging experience.",
# 		"Siri voice assistance upgrades: Siri now supports more languages and dialects, and offers a more conversational interaction style, making it easier to communicate and receive assistance.",
# 		"Apple Maps 3D view: The revamped Apple Maps now includes a 3D view option, allowing users to explore cities and landmarks in a more immersive and realistic way.",
#     ],
#     "Known Issues": [
#         "Devices with a large number of installed apps will display an Apple logo with progress bar for an extended period while the file system format is updated. This is a one-time migration when upgrading to iOS 17 beta for the first time.",
#         "MP3 files with malformed ID3 tags will fail to play.",
# 		"With any Classroom class set up, the AirDrop browser on teacher and student devices will not show any device"
#     ]
# }
# release_note = json.dumps(release_note)
# release_dict = {
#             "repo_name": "ios",
#             "release_version": "v17.0",
#             "release_note": release_note
# }
#
# df = pd.DataFrame(data=[release_dict])
# pg_service.handle_table(data_df=df)

# def get_release():
#     git_conn = GithubClient().git_conn()
#     svc = GithubService(github=git_conn)
#     repository = svc.get_repo("hwchase17/langchain")
#     releases = GithubService.get_releases(repository)
#     if len(releases) > 1:
#         end_time = releases[1].published_at
#     else:
#         end_time = ""
#     pulls = GithubService.get_repo_pulls(repository, end_time)
#     issues = GithubService.get_issues(repository)
#     rst = GithubService.release_notes_content(
#         repo_name="hwchase17/langchain",
#         releases=releases,
#         pulls=pulls,
#         issues=issues,
#         table_name="repo_release_contents")
#     print(rst)
#
# get_release()
import dotenv
import os
from modules.model import get_embedding, get_llm
import json
from langchain.vectorstores import Milvus
from langchain.docstore.document import Document
from components.count_tokens import keys_to_extract
from modules.chat_with_json import JsonAgents
from modules.mass_milvus import MassMilvus
from fuzzywuzzy import fuzz
dotenv.load_dotenv()

# set up environment variables
MILVUS_HOST = os.environ['MILVUS_HOST']
OPENAI_API_KEY = ""
OPENAI_API_BASE = os.environ['OPENAI_API_BASE']
OPENAI_API_TYPE = os.environ['OPENAI_API_TYPE']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
# question = "langchain有什么主要的模块"
# vector_collection = "likes_answers_collection"
# git_instance = ChatService.get_milvus_instance(collection_name=vector_collection)
# docs = asyncio.run(GithubAgent.get_docs(query=question,
#                                         git_instance=git_instance,
#                                         k=3))
# CollectAnswers.collect_search(docs=docs,
#                               table_name="likes_answers_collection")
# from components.count_tokens import CountTokens
#
# with open("D:\WorkSpace\streamlit_auth\example\JSON\\Input_Text.json") as f:
#     data = f.read()
# num = CountTokens.num_tokens_from_text(data)
# print(num)

folder_path = "D:\\WorkSpace\\streamlit_auth\\example\\JSON"
docs = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    component_type = filename.split(".")[0]
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        extract_data = JsonAgents.extract_keys(data=data, keys_to_extract=keys_to_extract)
        extract_data = json.dumps(extract_data)
        print(extract_data)
    metadata = {
        "type": component_type
    }
    doc = Document(page_content=extract_data, metadata=metadata)
    docs.append(doc)
# MassMilvus.afrom_documents(
#     docs,
#     collection_name="dev_figma_components",
#     connection_args={"host": MILVUS_HOST}
# )
# file_path = "D:\\WorkSpace\\streamlit_auth\\example\\JSON\\Button_Secondary.json"
#
# def convert_to_percentage(probability):
#     return "{:.1%}".format(probability)
#
# with open(file_path, 'r', encoding="utf-8") as file:
#     data = json.load(file)
#     extract_data = JsonAgents.extract_keys(data=data, keys_to_extract=keys_to_extract)
#     extract_data = json.dumps(extract_data)
#
# print(extract_data)
# embeddings = get_embedding("Ada-embedding")
# instance = Milvus(
#     collection_name="dev_figma_components",
#     embedding_function=embeddings,
#     connection_args={"host": MILVUS_HOST},
# )
# l1 = list()
# collect_docs = instance.similarity_search_with_score(extract_data)
# print(collect_docs)
# for doc in collect_docs:
#     types = doc[0].metadata["type"]
#     probability = (1 - (round(doc[1], 3) / 0.5))
#     l1.append({
#         "Type": types,
#         "Probability": "{:.1%}".format(probability)
#     })
# print(l1)

