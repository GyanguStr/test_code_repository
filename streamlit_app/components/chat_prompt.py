import re
from typing import List, Union
from langchain.agents import (
    Tool,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish


class ChatPrompt:
    """This class defines that the prompt for chatgpt"""

    query_prompt_summaries = """Rephrase the following question into a declarative sentence.
    Just copy the question and paste it into the "Rephrased sentence" section if the question is already clear and short enough.    
    Don't include any punctuations in the final rephrased sentence.
    Capture the main idea of the question in the final rephrased sentence.
    Refer the following examples for the rephrasing.
    EXAMPLE:
    1. "markdown bolt" -> "markdown bolt"
    2. "how can I learn the artificial intelligence in the most efficient way?" -> "the most efficient way to learn artificial intelligence"
    3. "I wonder if Grenoble is the largest city in France?" -> "largest city France Grenoble"
    The final rephrased question must combine the current question in the "Question" section and the history of the conversation between the user and the chatbot in the "History" section.
    If the History section is empty, ignore the history's influence on the rephrased sentence.
    Make sure your sentence can be understood by a 10-year-old child.
    The final rephrased sentence should be no longer than 15 words and must in english.

    Question: {query}

    History: {history}

    Rephrased sentence:"""

    prompt_template = """You are an intelligent assistant helping people solve the problems they ask.
    Given the following a question from human and several extracted parts of a long document.
    You can supplement and improve the reply based on the history of the conversation and the extracted parts of the document.
    Please do not repeat the content from HISTORY as an answer. 
    The content in HISTORY is for you to understand the context of the conversation. You should rely on the information from the Sources section
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    

    QUESTION: {question}
    =========
    Sources: {summaries}
    =========
    HISTORY: {chat_history}
    =========
    ANSWER: """

    agent_template = """Answer the following questions as best you can, You must use the following tools.
    After you decide which tool to use, copy the "Question" to the "Action Input" without making any changes.
    There are keywords "release" or "releases" in "Question", choose from the other available tools and do not use the "Common chat" tool.

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Please remember to use the same language as the Question when giving the final answer.

    Question: {input}
    {agent_scratchpad}"""

    pr_summaries = """The following content is obtained from a certain GitHub repository, 
    which includes all the modified files and their changes between two releases. 
    The content after the "+" symbol represents the added content, while the content after the "-" symbol represents the deleted content. 
    Generate a summary based on these changes. 
    
    {text}
    """

    release_prompt = """
    Below is a set of data provided for you. 
    Based on the content provided in the data, generate a release notes with the following format:
    Please return the output format in the following dictionary form.
    Below is an example of For example, you can refer to it, but don't use the content in it.
    with the content in the parentheses of the value as a hint, describing what aspect the content at the value is about. 
    Do not include the hint in the final output.
        "Overview": (Provide an overview of the release notes),
        “Release version”: (Provide version of the release notes, contents based on release_version),
        "Repository name": (Provide name based on repo_name),
        "What's Changed": [XXXXX, XXXXX, XXXXX](contents based on release_body and release_change_contents),
        "New Features": [XXXXX, XXXXX, XXXXX](List all new features introduced in this release),
        "Known Issues": [XXXXX, XXXXX, XXXXX](Describe known issues)
    The content of "New Features" mainly summarizes and describes the newly added functions of this release, do not repeat the content of "What's Changed".
    You can also further improve the release notes based on the provided data,but the return format must conform to the dictionary format.
    but if the data is not sufficient for you to write a release notes, you should say you don't know and cannot make up content.
    
    For example, the following content should be returned in dictionary format:
            "Overview": "This release includes various updates and additions to the XXXXX repository.",
            "What's Changed": [
                "Initial Streamlit callback integration doc (md)",
                "Zapier update oauth support",
            ],
            "New Features": [
                "Initial Streamlit callback integration doc (md)",
                "Zapier update oauth support",
            ],
            "Known Issues": [
                "sqldatabasechain result incomplete",
                "Different results when loading Chroma() vs Chroma.from_documents",
            ]
    QUESTION: {question}
    =========
    Data: {text}
    """

    db_release_prompt = """Given an input question, first create a syntactically correct {dialect} query to run, 
    then look at the query results and return answers.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    Only use the following tables and repo_name:

    {table_info}
    The following is an explanation of each field in the table. Combine the fields you think are most useful based on the question to obtain the most useful data and make your best answer.

    "repo_name": As the name suggests, the name of the github library
    "release_version": As the name suggests, the version number of the release
    "release_date": The specific time when the corresponding release version was released
    "release_body": The basic information of this release, combined with the content in release_change_contents, should be able to answer some questions about the new additions and changes in the current release.
    "release_change_contents": As the name suggests, the main changes in this release may include some new features and bug fixes
    "issues_contents": As the name suggests, some issues that are known but not yet resolved in the current release

    Question: {input}"""

    json_re_suffix = """
    This JSON data describes a graphical interface component from the design tool Figma. 
    You need to use your knowledge base to understand the meaning of each field in Figma and combine the field meanings with the data provided in the JSON to accurately answer the user's question. 

    Here are a few examples of field names and their meanings:
        fills: The fill style of the element, including color, gradient, or image. There can be multiple fill styles. 
        cornerRadius: The corner radius of a rounded rectangle.
        children: Usually represents the data of nested layers, including all child layers or components under the current layer or component. Each child layer or component has its own properties and styles, and may have their own "children", forming a nested structure. 


    This example is a combination of field meanings, JSON data, and user questions. This is just an example for your reference.
        Question: What does this JSON describe? 
        Answer: It represents a green rounded button with the text "Click Me" on it, using the Inter Semi Bold font, with a font size of 16, and white text color.

    You have to interpret each field like above and then you have to combine the json data and the "Question" to give an exact answer
    Note: The content in "HISTORY" is for you to understand the context of the conversation.
    
    Begin!

    Question: {input}
    Thought: I should look at the keys that exist in data to see what I have access to
    {agent_scratchpad}
    
    HISTORY: {chat_history}
    """

    json_query_prompt_summaries = """Rephrase the following question into a declarative sentence.
    Just copy the question and paste it into the "Rephrased sentence" section if the question is already clear and short enough.    
    Don't include any punctuations in the final rephrased sentence.
    Capture the main idea of the question in the final rephrased sentence.
    Refer the following examples for the rephrasing.
    EXAMPLE:
    1. "markdown bolt" -> "markdown bolt"
    2. "how can I learn the artificial intelligence in the most efficient way?" -> "the most efficient way to learn artificial intelligence"
    3. "I wonder if Grenoble is the largest city in France?" -> "largest city France Grenoble"
    The final rephrased question must combine the current question in the "Question" section and the history of the conversation between the user and the chatbot in the "History" section.
    If the History section is empty, ignore the history's influence on the rephrased sentence.
    Make sure your sentence can be understood by a 10-year-old child.
    The final rephrased sentence should be no longer than 15 words.
    
    Note: Must be consistent with the language used in human Question.

    Question: {query}

    History: {chat_history}

    Rephrased sentence:"""

    json_agent_template = """Answer the following questions as best you can, You must use the following tools.
       After you decide which tool to use, MUST copy the "Question" to the "Action Input" without making any changes.

       {tools}

       Use the following format:

       Question: the input question you must answer
       Thought: you should always think about what to do
       Action: the action to take, should be one of [{tool_names}]
       Action Input: the input to the action
       Observation: the result of the action
       ... (this Thought/Action/Action Input/Observation can repeat N times)
       Thought: I now know the final answer
       Final Answer: the final answer to the original input question

       Begin! Please remember to use the same language as the Question when giving the final answer.

       Question: {input}
       {agent_scratchpad}"""

    json_overall_prompt = """I want you to act as a proficient UI/UX designer in Figma. 
    I will provide you with a JSON file, which is an exported component from Figma.  
    Your task is to use your professional knowledge base to understand the meaning of each field in Figma and combine the field meanings with the data provided in the JSON to accurately answer the user's question. 

    Note: The content in "HISTORY" is for you to understand the context of the conversation. You should rely on the information from the "Data" section

    Choose one of the following two answering methods according to the amount of answering content
        If your answer is more content, use the format below:
            1. XXXXXXXXXX.\n
            2. XXXXXXXXXX.\n
            3. XXXXXXXXXX.\n
            ... (Not just three, but more according to your needs)

        If your answer is relatively short, you can express it clearly in one sentence.

    Begin, let's think step by step!!!


    Question: {query}
    =================
    Data: {data}
    =================
    HISTORY: {chat_history}
    """

    json_identify_prompt = """I want you to act as a proficient UI/UX designer in Figma. 
    I will provide you with a JSON file, which is an exported component from Figma. 
    Your task is to use your professional knowledge to analyze this JSON step by step, to determine what style the component represented by this JSON is, 
    such as whether this component is a button, an input box, a status bar, a checkmark, ..., etc. 
    If this component could be either a button or an input box, you need to analyze step by step which one is more likely,
    and what the probability is, referring to the example.
    
    After you identify the style of this component, you need to return the answer in the following JSON format. 
    
    Example: 
        Question: What style do you think this component is? 
        Thought: The name of the component is 'Checkbox/Empty', which directly means that it is a checkbox and contains a 'rectangle' type subkey named 'BG', which is usually used as the background for the checkbox, and an 'instance' type subkey called 'Icon/Check Mark', which is usually used as a check mark for the checkbox.
        Answer: 
            "Type": "Checkbox", 
            "Probability": "100%", 
            "Reasons": ["xxxxxxxx", "xxxxxxxx", ...]
            
        (**The above content must use JSON format**)

        Question: Based on your judgment, what could this component be? 
        Thought: This component has two "TEXT" type sub-items, usually a button only has one clearly expressed sub-item, so the likelihood of this component being an input box is higher than that of a button. The width of this component is >300, which is too wide for a button. Usually, the width of a button is between 100 and 150, so the probability of this component being an input box is higher.
        Answer: 
            "Type": "input box", 
            "Probability": "75%", 
            "Reasons": ["xxxxxxxx", "xxxxxxxx", ...]
            
        (**The above content must use JSON format**)
    
    Begin, let's think step by step!!!
    
    
    =================
    Question: {query}
    =================
    Data: {data}
    """


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        print(kwargs)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
