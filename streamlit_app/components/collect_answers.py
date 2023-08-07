import os
import uuid
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Tuple
from components.postgre_wrapper import PgService
from modules.mass_milvus import MassMilvus
from langchain.docstore.document import Document


@dataclass
class CollectAnswers:
    """The class that collects high-quality answers from bots"""

    pg_table_name: str
    vector_collection: str = "likes_answers_collection"
    vector_host: str = (
        os.getenv("MILVUS_HOST") if os.getenv("MILVUS_HOST") else ""
    )
    pg_service: PgService = field(init=False)

    def __post_init__(self):
        self.pg_service = PgService(table_name=self.pg_table_name)

    """Milvus: uuid, human_query; PG: uuid + human_query + bot_answers + likes"""

    def collect_answers(self, collect_search: tuple, index: int, human_query: str, bot_answers: str, user_name: str,
                        thumbs: str = ""):
        if collect_search and thumbs:
            update_sql = ""
            update_id = collect_search[0]
            likes = int(collect_search[-1])
            if thumbs == "up":
                likes += 1
                update_sql = f"""
                            UPDATE public.likes_answers_collection
                            SET likes='{likes}', likes_user = array_append(likes_user, '{user_name}')
                            WHERE answers_id='{update_id}' AND NOT ('{user_name}' = ANY(likes_user));
                            """
            elif thumbs == "down" and likes > 0:
                likes -= 1
                update_sql = f"""
                            UPDATE public.likes_answers_collection
                            SET likes='{likes}', likes_user = array_remove(likes_user, '{user_name}')
                            WHERE answers_id='{update_id}' AND ('{user_name}' = ANY(likes_user));
                            """
            print("collect_answers:", index, thumbs)
            print(update_sql)
            if update_sql:
                self.pg_service.execute_sql(command=update_sql)
                search_sql = f"""
                             SELECT answers_id, human_query, bot_answers, collect_user, likes FROM likes_answers_collection
                             WHERE answers_id = '{update_id}';
                             """
                rst = self.pg_service.execute_sql(command=search_sql)
                st.session_state["collect_search"][index] = rst[0]
        elif not collect_search and not thumbs:
            answers_id = str(uuid.uuid4())
            self.collect_milvus_postgresql(answers_id=answers_id,
                                           human_query=human_query,
                                           bot_answers=bot_answers,
                                           user_name=user_name,
                                           index=index)

    def collect_milvus_postgresql(self, answers_id: str, human_query: str, bot_answers: str, user_name: str,
                                  index: int):
        print(answers_id)
        vector_contents = Document(
            page_content=human_query,
            metadata={"answers_id": answers_id}
        )

        MassMilvus.afrom_documents(
            [vector_contents],
            collection_name=self.vector_collection,
            connection_args={"host": self.vector_host}
        )
        pg_contents = {
            "answers_id": answers_id,
            "human_query": human_query,
            "bot_answers": bot_answers,
            "collect_user": user_name,
            "likes": 0,
            "likes_user": f"{{}}"
        }
        df = pd.DataFrame(data=[pg_contents])
        self.pg_service.handle_table(data_df=df)
        st.session_state["collect_search"][index] = (answers_id,
                                                     human_query,
                                                     bot_answers,
                                                     user_name,
                                                     0)
        # print(st.session_state["collect_search"])

    @staticmethod
    def collect_search(table_name: str, docs: List[List[Tuple[Document, float]]]):
        answers_info = list()
        sorted_data = list()
        if docs:
            for doc in docs[0]:
                answers_id = doc[0].metadata["answers_id"]
                score = doc[1]
                if score <= 0.1:
                    answers_info.append(
                        {
                            "answers_id": answers_id,
                            "score": score
                        }
                    )
        if answers_info:
            min_score_answers = answers_info[0]["answers_id"]
            pg_service = PgService(table_name=table_name)
            search_sql = f"""
                         SELECT answers_id, human_query, bot_answers, collect_user, likes FROM likes_answers_collection
                         WHERE answers_id IN ({", ".join(["'" + answer["answers_id"] + "'" for answer in answers_info])});
                         """
            rst = pg_service.execute_sql(command=search_sql)

            sorted_data = sorted(rst, key=lambda x: int(x[-1]), reverse=True)
        if sorted_data:
            result = [item for item in sorted_data if item[0] == min_score_answers]
            similar_result = [item for item in sorted_data if item[0] != min_score_answers]
            st.session_state["collect_search"] = result + [tuple()] + st.session_state.get("collect_search", [])
            if similar_result:
                st.session_state["git_near_answers"] = similar_result
        else:
            st.session_state["collect_search"] = [tuple()] + [tuple()] + st.session_state.get("collect_search",
                                                                                              [])
        print(st.session_state["collect_search"])
        return sorted_data

    @staticmethod
    def collect_history_conversation(table_name: str, user_name: str):
        pg_service = PgService(table_name=table_name)
        search_sql = f"""
                     SELECT human_query, bot_answers
                        FROM public.likes_answers_collection
                        WHERE collect_user = '{user_name}'
                        ORDER BY "timestamp" DESC
                        LIMIT 5;
                     """
        rst = pg_service.execute_sql(command=search_sql)
        return rst
