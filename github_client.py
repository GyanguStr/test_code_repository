from github import Github
from dataclasses import dataclass, field
from modules.chat_with_git import ChatService



g = Github(login_or_token="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")
repo = g.get_repo("theskumar/python-dotenv")
rst = ChatService.get_release_notes(repo_name="theskumar/python-dotenv", table_name="repo_release_contents")
print(rst)


# releases = repo.get_releases()
# # for r in releases:
# #     print(r.title, r.tag_name, r.body)
# print(releases)
#
# latest_release = repo.get_latest_release()
#
# # 获取版本号、发布日期、发布说明
# version = latest_release.tag_name
# release_date = latest_release.published_at
# release_notes = latest_release.body
#
# # 输出获取到的信息
# print("版本号:", version, type(version))
# print("发布日期:", release_date, type(release_date))
# print("发布说明:", release_notes, type(release_notes))
# open_issues = repo.get_issues(state="open")
#
# print("\n已知问题：")
# for issue in open_issues:
#     print(f"- {issue.title}")

# 获取两次发布之间的所有 PR
# def get_prs_between_releases(repo, release1, release2):
#     prs = []
#     pulls = repo.get_pulls(state='closed', sort='created', direction='desc')
#     for pull in pulls:
#         p = pull.get_files()
#     for pr in repo.get_pulls(state='closed', sort='created', direction='desc'):
#         print(release1.created_at, release2.created_at, pr.merged_at)
#         if pr.merged_at:
#             if release1.created_at < pr.merged_at < release2.created_at:
#                 prs.append(pr)
#     return prs
#     # print(prs[0].merged_at, type(pulls[0].merged_at))
#     # return pulls[0]
#
# # 获取两次发布之间的所有文件更改
# def get_file_changes_between_releases(repo, release1, release2):
#     file_changes = []
#     prs = get_prs_between_releases(repo, release1, release2)
#     for pr in prs:
#         for file in pr.get_files():
#             file_changes.append(file)
#     return file_changes
#
# # 示例：获取第一个和第二个发布之间的所有 PR 和文件更改
# release1 = releases[0]
# release2 = releases[1]
# print(release1.title, release1.tag_name, release2.title, release2.tag_name)
#
# prs = get_prs_between_releases(repo, release1, release2)
# changes = prs.get_files()
# for i in changes:
#     print(i.filename, i.patch)
    # print(repo.get_contents(i.filename).decoded_content.decode())
# file_changes = get_file_changes_between_releases(repo, release1, release2)
#
# print("Pull Requests between releases:")
# for pr in prs:
#     print(pr.title)
#
# print("\nFile changes between releases:")
# for file_change in file_changes:
#     print(file_change.filename)



import base64

# 获取两次发布之间的所有文件更改及其完整内容
# def get_file_changes_and_full_content_between_releases(repo, release1, release2):
#     file_changes_and_full_content = []
#     prs = get_prs_between_releases(repo, release1, release2)
#     for pr in prs:
#         for file in pr.get_files():
#             file_content = repo.get_contents(file.filename)
#             decoded_content = base64.b64decode(file_content.content).decode('utf-8')
#             file_changes_and_full_content.append((file.filename, decoded_content))
#     return file_changes_and_full_content
#
# # 示例：获取第一个和第二个发布之间的所有文件更改及其完整内容
# file_changes_and_full_content = get_file_changes_and_full_content_between_releases(repo, release1, release2)
#
# print("File changes and full content between releases:")
# for file_change, full_content in file_changes_and_full_content:
#     print(f"Filename: {file_change}")
#     print(f"Full content:\n{full_content}\n")