import os
import langchain.memory.buffer
from streamlit import session_state
from psycopg2 import pool
from psycopg2.errors import UndefinedTable, ProgrammingError
import pandas as pd
from langchain.callbacks import openai_info
from functools import wraps
from components.key_vault import FetchKey
from typing import Callable
from dataclasses import dataclass, field
from io import StringIO
from modules.model import model_dict
from langchain.callbacks.openai_info import OpenAICallbackHandler


@dataclass
class PgService:
    """class with decorator functions for connecting to the database and inserting data"""
    host: str = (
        os.getenv("PG_HOST") if os.getenv("PG_HOST") else ""
    )
    dbname: str = (
        os.getenv("PG_DBNAME") if os.getenv("PG_DBNAME") else ""
    )
    user: str = (
        os.getenv("PG_USER") if os.getenv("PG_USER") else ""
    )
    ssl_mode: str = (
        os.getenv("SSL_MODE") if os.getenv("SSL_MODE") else ""
    )
    table_name: str = (
        os.getenv("TABLE_NAME") if os.getenv("TABLE_NAME") else ""
    )
    table_schema: str = (
        os.getenv("TABLE_SCHEMA") if os.getenv("TABLE_SCHEMA") else ""
    )
    password: str = FetchKey("PG-PASSWORD").retrieve_secret()
    conn_pool: pool.SimpleConnectionPool = field(init=False)

    def __post_init__(self):
        conn_string = f"host={self.host} user={self.user} dbname={self.dbname} password={self.password} sslmode={self.ssl_mode}"
        self.conn_pool = pool.SimpleConnectionPool(1, 500, conn_string)
        if self.conn_pool:
            print("Connection pool created successfully")
        else:
            print("Connection pool created fail")
            return False

    def insert_decorator(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            question = (*args,)
            openai_cb = func(*args, **kwargs)
            func_dict = {"question": question[0], "name": func.__name__}
            print(func_dict)
            df = PgService.process_data(openai_cb, func_dict)
            self.handle_table(df)

            return openai_cb

        return wrapper

    def handle_table(self, data_df: pd.DataFrame, clear=False):
        conn = self.conn_pool.getconn()
        cur = conn.cursor()
        columns = [each.lower() for each in data_df.columns.to_list()]
        print(columns)
        if_exist_table = f"select 1 from pg_tables where schemaname = '{self.table_schema}' and tablename = '{self.table_name}';"
        cur.execute(if_exist_table)
        has_table = cur.fetchall()
        if not has_table:
            columns_sql = [f"{each} VARCHAR" for each in columns]
            columns_sql = ",".join(columns_sql)
            create_table_sql = f"""
                    CREATE TABLE {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    {columns_sql},
                    timestamp varchar(40) DEFAULT (now())
                    )
                    """
            cur.execute(create_table_sql)
            conn.commit()
            print(f"{create_table_sql}, creating table {self.table_name} succeed!")
        else:
            print(f"table {self.table_name} already exist!")
        if clear:
            truncate_sql = f"TRUNCATE {self.table_name} RESTART IDENTITY;"
            cur.execute(truncate_sql)
            conn.commit()
        dict_list = data_df.to_dict(orient="records")
        insert_sql = f"""
                   INSERT INTO  {self.table_name} 
                   ({", ".join([col for col in columns])})
                   VALUES 
                   ({", ".join(["%s" for _ in columns])})
                    """
        print(insert_sql)
        for each in dict_list:
            value = tuple(each.values())
            insert_sql = insert_sql
            cur.execute(insert_sql, value)
            conn.commit()
            print(f"Succeed to push chat log in pg table: {self.table_name}")
        cur.close()
        conn.close()

    def handle_df_table(self, data_df: pd.DataFrame, clear=False):
        conn = self.conn_pool.getconn()
        cur = conn.cursor()
        columns = data_df.columns.to_list()
        columns = (each.lower() for each in columns)
        if_exist_table = f"select 1 from pg_tables where schemaname = '{self.table_schema}' and tablename = '{self.table_name}';"
        cur.execute(if_exist_table)
        has_table = cur.fetchall()
        if not has_table:
            columns_sql = [f"{each} VARCHAR" for each in columns]
            columns_sql = ",".join(columns_sql)
            create_table_sql = f"""
                            CREATE TABLE {self.table_name} (
                            id SERIAL PRIMARY KEY,
                            {columns_sql},
                            timestamp varchar(40) DEFAULT (now())
                            )
                            """
            cur.execute(create_table_sql)
            conn.commit()
            print(f"{create_table_sql}, creating table {self.table_name} succeed!")
        else:
            print(f"table {self.table_name} already exist!")
        if clear:
            truncate_sql = f"TRUNCATE {self.table_name} RESTART IDENTITY;"
            cur.execute(truncate_sql)
            conn.commit()

        data_buffer = StringIO()
        data_df.to_csv(data_buffer, sep="^", index=False, header=False)
        pg_data = StringIO(data_buffer.getvalue())
        cur.copy_from(pg_data, self.table_name, null="", sep="^", columns=columns)
        conn.commit()
        print(f"Succeed to push data in pg table: {self.table_name}")
        cur.close()
        conn.close()

    def execute_sql(self, command: str):
        conn = self.conn_pool.getconn()
        if not command:
            print("no executable command .")
            return
        cur = conn.cursor()
        try:
            cur.execute(command)
        except UndefinedTable as e:
            print(e)
            return []
        try:
            rst = cur.fetchall()
        except ProgrammingError:
            rst = []
        conn.commit()
        cur.close()
        conn.close()
        return rst

    @staticmethod
    def process_data(openai_cb: OpenAICallbackHandler, func_dict: dict) -> pd.DataFrame:

        question = func_dict["question"]
        prompt = session_state.get(f"{func_dict['name']}_buffer")

        if isinstance(prompt, langchain.memory.buffer.ConversationBufferMemory):
            response = prompt.chat_memory.messages[-1].content

        elif isinstance(prompt, list):
            if not isinstance(prompt[-1], str):
                response = prompt[-1].content
            else:
                response = prompt[-1]

        model = model_dict[session_state.get("OPENAI_MODEL")]['model_name']
        chat_log = {
            "user_name": session_state.get("USER_INFO")["name"],
            "user_login_name": session_state.get("USER_INFO")["username"],
            "prompt": str(prompt),
            "question": question,
            "response": response,
            "model": model,
            "prompt_tokens": openai_cb.prompt_tokens,
            "response_tokens": openai_cb.completion_tokens,
            "total_tokens": openai_cb.total_tokens,
            "prompt_cost": openai_info.get_openai_token_cost_for_model(model,
                                                                       openai_cb.prompt_tokens),
            "response_cost": openai_info.get_openai_token_cost_for_model(model,
                                                                         openai_cb.completion_tokens,
                                                                         is_completion=True),
            "total_cost": openai_cb.total_cost
        }
        df = pd.DataFrame(data=[chat_log])
        return df

    @staticmethod
    def token_cost(num_tokens: int, model_name: str) -> float:
        model_cost_1k_tokens = {
            "text-embedding-ada-002": 0.0004,
        }
        if model_name not in model_cost_1k_tokens:
            pass

        return model_cost_1k_tokens[model_name] * num_tokens / 1000
