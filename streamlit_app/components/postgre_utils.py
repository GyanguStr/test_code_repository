import psycopg2
from psycopg2 import pool
import pandas as pd
from io import StringIO
import os
from langchain.callbacks import openai_info
from components.key_vault import FetchKey


class PostgreUtils:
    def __init__(self):
        # NOTE: fill in these variables for your own cluster
        self.host = "c.gpt-project-cosmos-db-postgresql.postgres.database.azure.com"
        self.dbname = "citus"
        self.user = "citus"
        self.password = FetchKey("PG-PASSWORD").retrieve_secret()
        self.sslmode = "require"
        self.conn = None
        self.cursor = None

    def build_connection(self):
        # Build a connection string from the variables
        conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(self.host, self.user, self.dbname,
                                                                                     self.password, self.sslmode)

        postgreSQL_pool = pool.SimpleConnectionPool(1, 20, conn_string)
        if (postgreSQL_pool):
            print("Connection pool created successfully")

        # Use getconn() to get a connection from the connection pool
        self.conn = postgreSQL_pool.getconn()

        self.cursor = self.conn.cursor()

    def close_connection(self):
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def drop_table(self, table_name):
        # Drop previous table of same name if one exists
        self.cursor.execute("DROP TABLE IF EXISTS {};".format(table_name))
        print("Finished dropping table (if existed)")

    # columns_str format: name type. example: column_name integer
    def create_table(self, table_name, columns_str: str):
        self.drop_table(table_name)
        # Create a table
        self.cursor.execute("CREATE TABLE {} ({})".format(table_name, columns_str))
        print("Finished creating table")

    def query(self, query_str):
        self.cursor.execute(query_str)
        result = self.cursor.fetchall()
        return result

    def insert(self, df: pd.DataFrame, table_name: str, columns: tuple):
        df_len = df.shape[0]
        temp = StringIO()
        df.to_csv(temp, sep="^", index=False, header=False)
        insert_data = StringIO(temp.getvalue())
        self.cursor.copy_from(insert_data, table_name, sep="^", null="", columns=columns)
        temp.close()
        insert_data.close()
        print("insert {} rows into {} successfully".format(df_len, table_name))

    def langchain_data_insert(self, cb: openai_info.OpenAICallbackHandler, user_info, question, prompt, response,
                              model):
        openai_log_list = []
        openai_log = {}
        openai_log['user_name'] = user_info['name']
        openai_log['user_login_name'] = user_info['preferred_username']
        openai_log['question'] = question
        openai_log['prompt'] = prompt
        openai_log['response'] = response
        openai_log['model'] = model
        openai_log['prompt_tokens'] = cb.prompt_tokens
        openai_log['response_tokens'] = cb.completion_tokens
        openai_log['total_tokens'] = cb.total_tokens
        openai_log['prompt_cost'] = openai_info.get_openai_token_cost_for_model(model, cb.prompt_tokens)
        openai_log['response_cost'] = openai_info.get_openai_token_cost_for_model(model, cb.completion_tokens,
                                                                                  is_completion=True)
        openai_log['total_cost'] = cb.total_cost

        openai_log_list.append(openai_log)
        openai_log_df = pd.DataFrame(openai_log_list)
        self.build_connection()
        self.insert(openai_log_df, 'openai_log', tuple(openai_log_df.columns.tolist()))
        self.close_connection()


if __name__ == '__main__':
    a = PostgreUtils()
    a.build_connection()
    b = pd.read_csv('test.csv')
    a.insert(b, 'openai_log',
             ('question', 'response', 'question_tokens', 'response_tokens', 'total_tokens', 'total_cost'))
    a.close_connection()
