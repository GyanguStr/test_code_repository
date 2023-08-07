from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from components.key_vault import FetchKey


model_dict = {
    'Ada': {'deployment_name': 'ada', 'model_name': 'text-ada-001'},
    'Davinci': {'deployment_name': 'davinci-text', 'model_name': 'text-davinci-003'},
    'GPT-3': {'deployment_name': 'chat', 'model_name': 'gpt-3.5-turbo'},
    'GPT-3-16K': {'deployment_name': 'gpt-35-turbo-16', 'model_name': 'gpt-3.5-turbo-16k'},
    'GPT-4': {'deployment_name': 'chatgpt-4', 'model_name': 'gpt-4'},
    'GPT-4-32K': {'deployment_name': 'gpt-4', 'model_name': 'gpt-4-32k'},
    'Davinci-code': {'deployment_name': 'davinci-code', 'model_name': 'code-davinci-002'},
    'Ada-embedding': {'deployment_name': 'ada-embedding', 'model_name': 'text-embedding-ada-002'},
}

def azure_openai(model, temperature):
    llm = AzureOpenAI(
        temperature=temperature,
        deployment_name=model_dict[model]['deployment_name'],
        model=model_dict[model]['model_name'],
    )
    return llm

def chat_openai(model, temperature):
    chat = AzureChatOpenAI(
        temperature=temperature,
        deployment_name=model_dict[model]['deployment_name'],
        model=model_dict[model]['model_name'],
    )
    return chat

def get_llm(model, temperature):
    if model == 'GPT-3' or model == 'GPT-3-16K' or model == 'GPT-4' or model == 'GPT-4-32K':
        return chat_openai(model, temperature)
    else:
        return azure_openai(model, temperature)
    
def get_embedding(model: str = 'Ada-embedding'):
    embeddings = OpenAIEmbeddings(
        model=model_dict[model]['model_name'],
        deployment=model_dict[model]['deployment_name'],
        disallowed_special=()
    )
    return embeddings
