from typing import Optional
import psycopg2
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, exceptions
from modules.embedding import openai_embedding_call


class PostgreSQL:
    def __init__(self) -> None:
        self.conn = self.connect()
        self.cursor = self.conn.cursor()

    def connect(self):
        try:
            connection = psycopg2.connect(user="citus",
                                        password="Beacon2023",
                                        host="c.gpt-project-cosmos-db-postgresql.postgres.database.azure.com",
                                        port="5432",
                                        database="citus")
            return connection
        except Exception as e:
            raise e
        
    def insert_with_return(self, query):
        try:
            self.cursor.execute(query)
            self.conn.commit()
            return self.cursor.fetchall()
        except Exception as e:
            raise e


class PostgreSQLConversationTracker:

    def __init__(self) -> None:
        self.pg = PostgreSQL()

    def store_to_conversation_tracker(self, user_name: str):


        query = f"""INSERT INTO conversation_tracker (user_name)
        VALUES ('{user_name}') RETURNING id;"""

        conversation_tracker_id = self.pg.insert_with_return(query)[0][0]

        return conversation_tracker_id

    def get_conversation_tracker_id(self, user_name: str, return_limit: int = 100):

        query = f"""SELECT id, created_at FROM conversation_tracker
        WHERE user_name = '{user_name}';"""

        conversation_tracker_id = self.pg.insert_with_return(query)
        conversation_tracker_id = sorted(conversation_tracker_id, key=lambda x: x[1])

        output_without_duplicates: list = []

        count = 0
        while len(output_without_duplicates) < return_limit and count < len(conversation_tracker_id):
            if conversation_tracker_id[count][0] not in output_without_duplicates:
                output_without_duplicates.append(conversation_tracker_id[count][0])
            count += 1

        return output_without_duplicates


class MilvusDBConversationTracker:
    def __init__(self, 
        host: str = 'default', 
        port: str = 'default',
    ) -> None:
        self.host = host if host != 'default' else '52.226.226.29'
        self.port = port if port != 'default' else '19530'
        self.conn = self.connect()
    
    def connect(self):
        try:
            connections.connect(host=self.host, port=self.port)
            return
        except Exception as e:
            raise e
        
    def get_conversation_history(self, ids: list[int], collection_name: str = 'conversation_tracker', limit: int = 5):

        field_name = ['text', 'conversation_tacker_id', 'question', 'answer', 'created_at']
        output_res: list = []
        
        try:
            id_length = len(ids)
            count = 0
            id_diff = 0
            collection = Collection(collection_name)
            while id_diff < limit and id_length > count:
                try:
                    res = collection.query(
                        expr = f'conversation_tacker_id == {str(ids[count])}',
                        offset = 0,
                        limit = limit, 
                        output_fields = field_name,
                        consistency_level="Strong"
                    )
                    if res: 
                        output_res += res
                        id_diff += 1
                except exceptions.MilvusException as e:
                    pass
                
                count += 1
        except exceptions.SchemaNotReadyException:
            pass

        return output_res
    
    def get_similar_qa(self, query: str , collection_name: str = 'conversation_tracker', threshold: float = 0.1):
        output_fields = ['text', 'question', 'answer', 'created_at', 'conversation_tacker_id', 'private', 'pk']

        try:
            collection = Collection(name=collection_name)
        except Exception as e:
            return '', ''

        embedding = openai_embedding_call(query)
        emb = embedding['data'][0]['embedding']
        raw_cost = embedding['usage']

        cost = {
            'model': 'ada-embedding',
            'cost': round(raw_cost['total_tokens'] * 0.0004 / 1000, 6),
            'total_tokens': raw_cost['total_tokens'],
            'prompt': raw_cost['prompt_tokens'],
            'completion': 0,
        }

        # ===== Temporary param =====
        param = {"metric_type": "L2", "params": {"ef": 10}}

        results = collection.search(
            data=[emb], 
            anns_field="vector", 
            param=param,
            limit=10, 
            expr=None,
            # set the names of the fields you want to retrieve from the search result.
            output_fields=output_fields,
            consistency_level="Strong"
        )

        similar_response: str = ''

        for i, d in enumerate(results[0].distances):
            if d < threshold and results[0][i].entity.get('private') == False:
                similar_response = results[0][i].entity.get('answer')
                break

        return similar_response, cost




if __name__ == '__main__':
    mct = MilvusDBConversationTracker()
    x = mct.get_conversation_history([17, 15])
    for xx in x:
        print(xx['conversation_tacker_id'])
    print(len(x))
    