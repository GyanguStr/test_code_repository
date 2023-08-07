import streamlit as st
from modules.csv_reader import CSVLoader
from components.key_vault import FetchKey
from io import StringIO
import streamlit.components.v1 as components
import openai
from components.html_templates import light_user_template, old_light_bot_template

ss = st.session_state

# --- BACKEND ---
def get_completion(input, engine="chatgpt-4", temperature=0, detailed=False, one_shot=True): 
    openai.api_key = FetchKey("OPENAI-KEY").retrieve_secret()

    if one_shot:
        messages = [{"role": "user", "content": input}]
    else:
        messages = input

    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=temperature, 
    )

    if detailed:
        return response
    else:
        return response.choices[0].message["content"]

@st.cache_data
def convert_csv_object_to_table(csv_object) -> str:
    loader = CSVLoader(csv_object)
    data = loader.load()
    table_template = """ROW CONTENT: {row_content}
    SOURCE: {source}
    ROW NUMBER: {row_number}

    """
    table_context = [table_template.format(row_content=row.page_content, source=row.metadata['source'], row_number=row.metadata['row']) for row in data]
    return table_context

@st.cache_data
def convert_raw_table_to_markdown_table(table: str) -> str:
    PROMPT = """Translate the following table delimited with triple backticks into a markdown table.
    Do not include any other elements in your response except the markdown table.
    Do not include the triple backticks in your response.
    All content in status column must in lower case.
    Replace the term "in-progress" with "active" in the table.
    Remove only the term "backlog" in the table without affecting the whole row in the table.

    ```{table}```
    """
    PROMPT = PROMPT.format(table=table)
    response = get_completion(PROMPT)
    return response

@st.cache_data
def convert_markdown_table_to_mermaid_code(markdown_table: str) -> str:
    PROMPT = """Translate the following markdown table delimited with triple backticks into a mermaid code of gantt diagram.
    Use the "Task" column as the task name instead of "Abbr".
    Use the "Abbr" column after the term "after" in the mermaid code instead of "Task".
    The order of descrtiption after the colon must be: 1. Status, 2. Abbr, 3. After or Start date, 4.Duration
    FOR EXAMPLE: 
    1. Important task   :active, XX, 2023-06-07, 2w
    The mermaid code must contain status of the tasks in the "Status" column if the status is not empty.
    Give your answer just in format of the Gantt chart code beginning with "gantt" in the first line.
    Do not include the triple backticks in your response.

    ```{markdown_table}```
    """
    PROMPT = PROMPT.format(markdown_table=markdown_table)
    response = get_completion(PROMPT)
    return response

def convert_csv_to_mermaid_code(table_context: str) -> str:
    PROMPT = """Translate the following table which is delimited with triple backticks into a Gantt chart code:
    ```{table}```
    Use the Task column in the table as the task name instead of the Abbr column in the table.
    The order of descrtiption after the colon must be: 1. Status, 2. Abbr, 3. After or Start date, 4.Duration
    FOR EXAMPLE: 
    1. Important task   :active, XX, 2023-06-07, 2w
    Remove the term "backlog" in the code.
    Use the term "active" to indicate the status "in-progress".
    Give your answer just in format of the Gantt chart code beginning with "gantt" in the first line.
    Give your answer without any backticks.
    """
    PROMPT = PROMPT.format(table=table_context)
    response = get_completion(PROMPT)
    return response

@st.cache_data
def define_q_type(question: str) -> str:
    PROMPT = f"""Analyze the question from the user, classify the question into one of the following categories:
    1. Request to modify the table (output "request" in this case)
    2. Just a question (output "question" in this case)

    QUESTION: {question}
    """
    response = get_completion(PROMPT)
    return response

def update_cost_list(response: openai.openai_object.OpenAIObject, cost_name: str) -> None:
    model = response.model
    usage = response.usage
    completion = usage['completion_tokens'] * 0.12 / 1000
    prompt = usage['prompt_tokens'] * 0.06 / 1000
    cost = completion + prompt

    cost = {
        'model': model,
        'cost': cost,
        'total_tokens': usage['total_tokens'],
        'prompt': usage['prompt_tokens'],
        'completion': usage['completion_tokens'],
    }

    ss[cost_name] = [cost] + [{}] + ss.get(cost_name, [])


@st.cache_data
def answer_question(question: str, pm_markdown_table_name: str, ei_markdown_table_name: str, history_name: str, cost_name: str, roadmap_code_name: str, roadmap_history_name: str) -> None:
    PROMPT = f"""You are an assistant of a project manager helping him to arrange the tasks in various projects.
    You will be given a table in markdown table format that describes projects' planning in the PROJECT MANAGEMENT TABLE section.
    You will be given a table that describes the specialties of each employee in the EMPLOYEE INFORMATION TABLE section.
    Some of the employees are missing in the table, just treat them as if they can do all the tasks.
    If the question from the user is not clear, you can ask the user for clarification.
    Be careful to the specialty of the employee and the availability of the employee, always modify the table in a way that the employee can do the task and the employee is available at that time.

    PROJECT MANAGEMENT TABLE: {ss[pm_markdown_table_name]}

    EMPLOYEE INFORMATION TABLE: {ss[ei_markdown_table_name]}

    Do not include any table element in your response."""

    system_message = {'role': 'system', 'content': PROMPT}

    if ss.get(history_name):
        assert ss[history_name][0]['role'] == 'system'
        ss[history_name][0] = system_message
    else:
        ss[history_name] = [system_message]

    user_message = {'role': 'user', 'content': question}
    ss[history_name].append(user_message)

    response = get_completion(ss[history_name], temperature=0, detailed=True, one_shot=False)
    ss[history_name].append({'role': 'system', 'content': response.choices[0].message["content"]})
    ss[roadmap_history_name] = [ss[roadmap_code_name]] + [''] + ss.get(roadmap_history_name, [])

    update_cost_list(response, cost_name)

@st.cache_data
def fulfill_request(question: str, pm_markdown_table_name: str, ei_markdown_table_name: str, history_name: str, cost_name: str, roadmap_code_name: str, roadmap_history_name: str) -> None:
    PROMPT = f"""You are an assistant of a project manager helping him to arrange the tasks in various projects.
    Some of the employees are missing in the EMPLOYEE INFORMATION TABLE, just treat them as if they can do all the tasks.
    If the question from the user is not clear, you can ask the user for clarification, and output an empty string in place of the table.
    If the question from the user is clear enough, modify the table based on the question from the user and various contexts.
    Calculate the exact date in the format "DD-MM-YYY" to be used in "start date" column in the table.
    Always tell the user reason why you modified the table in such a way in your response in less than 5 sentences.
    Be careful to the specialty of the employee and the availability of the employee, always modify the table in a way that the employee can do the task and the employee is available at that time.

    PROJECT MANAGEMENT TABLE: {ss[pm_markdown_table_name]}

    EMPLOYEE INFORMATION TABLE: {ss[ei_markdown_table_name]}

    Generate the new PROJECT MANAGEMENT TABLE only based the one provided by user.
    Use "***" to seperate your answer and the modified PROJECT MANAGEMENT TABLE."""

    system_message = {'role': 'system', 'content': PROMPT}

    if ss.get(history_name):
        assert ss[history_name][0]['role'] == 'system'
        ss[history_name][0] = system_message
    else:
        ss[history_name] = [system_message]

    user_message = {'role': 'user', 'content': question}
    ss[history_name].append(user_message)

    response = get_completion(ss[history_name], temperature=0, detailed=True, one_shot=False)

    response_content = response.choices[0].message["content"]

    mermaid_code: str = ''
    if "***" in response_content:
        communication, markdown_table = response_content.split("***")
        markdown_table = markdown_table.strip()

        if ss[pm_markdown_table_name] != markdown_table:
            ss[pm_markdown_table_name] = markdown_table
            convert_markdown_table_to_mermaid_code.clear()
            mermaid_code = convert_markdown_table_to_mermaid_code(ss[pm_markdown_table_name])
            ss[roadmap_code_name] = mermaid_code
            
    else:
        communication = response_content
        mermaid_code = ss[roadmap_code_name]
    
    ss[roadmap_history_name] = [mermaid_code] + [''] + ss.get(roadmap_history_name, [])
    ss[history_name].append({'role': 'assistant', 'content': communication})
    update_cost_list(response, cost_name)


def mermaid(code: str) -> None:
    components.html(f"""
    <div style="height: 100%;">
        <pre class="mermaid" style="height: 100%;"> {code} </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize ({{
                startOnLoad: true
            }});
        </script>
    </div> 
    """, height=800)


# --- FRONTEND ---
roadmap_code_name = 'roadmap_code'
initial_mermaid_code_name = 'initial_roadmap_code'
pm_md_table_name = 'pm_markdown_table'
initial_pm_md_table_name = 'initial_pm_markdown_table'
ei_md_table_name = 'ei_markdown_table'
history_name = 'chat_with_roadmap_history'
cost_name = 'chat_with_roadmap_cost'
roadmap_history_name = 'roadmap_history'

def initialize_session_state(pop_keys: list = []):
    for key in pop_keys:
        ss.pop(key, None)

def write_history(history_name: str, cost_name: str, roadmap_history_name: str) -> None:
    messages = ss.get(history_name, []).copy()
    cost = ss.get(cost_name, []).copy()

    messages.reverse()

    for i, msg in enumerate(messages[:-1]):
        if i % 2 == 0:
            st.write(old_light_bot_template.format(MSG=msg['content'], MODEL=cost[i]['model'], COST=round(cost[i]['cost'], 6), TOKENS_USED=cost[i]['total_tokens'], PROMPT=cost[i]['prompt'], COMPLETION=cost[i]['completion']), unsafe_allow_html=True)
            if i != 0:
                with st.expander('Mermaid chart', expanded=False):
                    mermaid(ss[roadmap_history_name][i])
        else:
            st.write(light_user_template.format(MSG=msg['content']), unsafe_allow_html=True)

def show_response(roadmap_code: str, pm_markdown_table: str):
    mermaid(roadmap_code)

    with st.expander('Mermaid code and table'):
        if roadmap_code:
            st.caption("Feel free to copy the code and paste it into an editor and display the gantt chart elsewhere.")
            st.code(f"""{roadmap_code}""", language='mermaid')
        if pm_markdown_table:
            st.write(pm_markdown_table)

def roadmap_input_area():
    with st.expander('File uploader'):
        st.write('**Upload a project management file in .csv format**')
        st.file_uploader("Upload a project management file in .csv format", key='project_management_file', type='csv', label_visibility='collapsed')
        st.write('**Upload an employee information file in .csv format**')
        st.file_uploader("Upload an employee information file in .csv format", key='employee_info_file', type='csv', label_visibility='collapsed')
        upload_col, initialize_col, _, _, _ = st.columns([1, 1, 5, 5, 5])
        with upload_col:
            st.button("Uplaod", type="primary", key='csv_upload_button')
        with initialize_col:
            st.button('Initialize', key='initialize_button')

    if ss.get('initialize_button'):
        ss[pm_md_table_name] = ss[initial_pm_md_table_name]
        ss[roadmap_code_name] = ss[initial_mermaid_code_name]

    if ss.get('project_management_file') and ss.get('csv_upload_button'):

        # process project management file
        pm_stringio = StringIO(ss.get('project_management_file').getvalue().decode("utf-8"))
        convert_csv_object_to_table.clear()
        pm_raw_table = convert_csv_object_to_table(pm_stringio)
        convert_raw_table_to_markdown_table.clear()
        pm_markdown_table = convert_raw_table_to_markdown_table(pm_raw_table)
        ss[pm_md_table_name] = pm_markdown_table.strip()
        ss[initial_pm_md_table_name] = ss[pm_md_table_name]

        # process employee information file
        if ss.get('employee_info_file'):
            ei_stringio = StringIO(ss.get('employee_info_file').getvalue().decode("utf-8"))
            convert_csv_object_to_table.clear()
            ei_raw_table = convert_csv_object_to_table(ei_stringio)
            convert_raw_table_to_markdown_table.clear()
            ei_markdown_table = convert_raw_table_to_markdown_table(ei_raw_table)
        else:
            ei_markdown_table = ''
        ss[ei_md_table_name] = ei_markdown_table.strip()

        convert_markdown_table_to_mermaid_code.clear()
        mermaid_code = convert_markdown_table_to_mermaid_code(pm_markdown_table)
        ss[roadmap_code_name] = mermaid_code
        ss[initial_mermaid_code_name] = ss[roadmap_code_name]
        

def roadmap_qa_area():
    if ss.get('pm_markdown_table'):
        st.text_input('Talk with your assistant', key='chat_with_roadmap_q')
        submit_col, clear_col, _, _, _ = st.columns([1, 1, 5, 5, 5])
        with submit_col:
            button_submit = st.button('Submit', type="primary")
        with clear_col:
            button_clear = st.button('Clear')

        if button_clear:
            pop_list = ['chat_with_roadmap_history', 'chat_with_roadmap_cost']
            initialize_session_state(pop_list)

        if button_submit and ss.get('chat_with_roadmap_q'):
            question = ss.get('chat_with_roadmap_q')
            define_q_type.clear()
            q_type = define_q_type(question)
            print(q_type)
            if q_type == 'question':
                answer_question.clear()
                answer_question(question, pm_md_table_name, ei_md_table_name, history_name, cost_name, roadmap_code_name, roadmap_history_name)
            elif q_type == 'request':
                fulfill_request.clear()
                fulfill_request(question, pm_md_table_name, ei_md_table_name, history_name, cost_name, roadmap_code_name, roadmap_history_name)

def roadmap():

    if ss.get(roadmap_code_name) or ss.get(pm_md_table_name):
        show_response(ss.get(roadmap_code_name), ss.get(pm_md_table_name))

    if ss.get(history_name):
        write_history(history_name, cost_name, roadmap_history_name)