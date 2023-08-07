from langchain.vectorstores import Milvus
import openai
import asyncio
import time
from components.key_vault import FetchKey


MILVUS_HOST='52.226.226.29'
milvus_connection_args = {'host': MILVUS_HOST}

openai.api_key = FetchKey("OPENAI-KEY").retrieve_secret()
openai.api_type = "azure"
openai.api_base = "https://mobile-beacon.openai.azure.com/"
openai.api_version = "2023-05-15"

def embed_with_azure_openai(doc, embeddings, collection_name='temp'):
    # adopt Azure OpenAI
    index = Milvus.from_documents(
        documents=doc,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=milvus_connection_args,   
    )
    return index

def openai_embedding_call(text: str, engine: str ='ada-embedding'):
    response = openai.Embedding.create(
        input=text,
        engine=engine,
        timeout=5,
    )

    return response

async def get_embeddings(test: str, engine: str ='ada-embedding'):
    response = await asyncio.to_thread(openai_embedding_call, test, engine)
    return response

def seperate_tasks(tasks, embeddings):
    done = []
    pending = []
    for task, e in zip(tasks, embeddings):
        if isinstance(e, Exception):
            pending.append(task.get_name())
        else:
            done.append({
                'name': task.get_name(),
                'result': e
            })
    return done, pending

async def aget_embeddings(chunks: list[str], retry: int =30, batch_size: int = 1000):

    print('Starting embedding...')

    ts = time.time()

    total_count = len(chunks)

    total_done : list = []

    for i in range(0, total_count, batch_size):

        # Grab end index
        end = min(i + batch_size, total_count)

        print(f'Processing: {i} - {end} (Total: {total_count})')

        # generate initial task list
        tasks = [asyncio.create_task(get_embeddings(chunks[i]), name=str(i)) for i in range(i, end)]

        retry_for_a_round = retry

        emb = await asyncio.gather(*tasks, return_exceptions=True)

        done, pending = seperate_tasks(tasks, emb)

        print(f' 1st try: done: {len(done)}, pending: {len(pending)}')

        while retry_for_a_round > 0 and any(isinstance(e, Exception) for e in emb):
            print(f' Retrying {len(pending)} tasks')
            retry_for_a_round -= 1
            tasks = [asyncio.create_task(get_embeddings(chunks[int(i)]), name=i) for i in pending]
            emb = await asyncio.gather(*tasks, return_exceptions=True)
            new_done, pending = seperate_tasks(tasks, emb)
            done += new_done

        if any(isinstance(e, Exception) for e in emb):
            raise Exception('Failed to get embeddings')
        else:
            print(f'Finished.')
            total_done += done
            
            
    total_done = sorted(total_done, key=lambda k: int(k['name']))
    total_done = [d['result'] for d in total_done]
            
    print(f'Embedding time taken: {time.time() - ts:.2f} seconds')
    return total_done

    