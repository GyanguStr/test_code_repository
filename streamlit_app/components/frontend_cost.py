import time
import streamlit as st
from langchain.callbacks.openai_info import OpenAICallbackHandler
from functools import wraps

ss = st.session_state

def get_cost_dict(cb: OpenAICallbackHandler) -> dict:
    cost = {
        'model': ss['OPENAI_MODEL'],
        'cost': cb.total_cost,
        'total_tokens': cb.total_tokens,
        'prompt': cb.prompt_tokens,
        'completion': cb.completion_tokens,
    }
    return cost


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Duration {func.__name__} - {round(end_time - start_time, 2)} seconds.")
        st.session_state[f"time_spent_{func.__name__}"] = [round(end_time - start_time, 2)] + \
                                                          [0.0] + \
                                                          st.session_state.get(f"time_spent_{func.__name__}", [])
        return result
    return wrapper
