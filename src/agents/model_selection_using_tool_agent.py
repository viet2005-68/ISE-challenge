import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from constants.constants import BASE_URL, LLAMA_MODEL
from ..state.agent_state import AgentState
from ..model.error import Error
from tools.tools import tools

llm = ChatOpenAI( 
    base_url=BASE_URL,
    model=LLAMA_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def model_selection_using_tool_agent(state: AgentState):
    system_prompt = """
                    You are a machine learning expert assigned to select the **Best-fit model** for a given task.
                    Given:
                    - A user task description
                    - A list of available models (names and url)
                    - A tool that fetches model details from a provided URL (it help gathers model description, input/output format, code sample usage)

                    Your job is to:
                    1. Use the **provided tool** to retrieve real details about each candidate model:
                        - ✅ Model description
                        - 📥 Input format
                        - 📤 Output format
                        - 🛠️ Library requirements
                        - 🧪 Code sample (usage code snippet)
                    2. Select the best model for user given task based on the data that the tool gives you.

                    ⚠️ VERY IMPORTANT RULES

                    - ❌ DO NOT assume or invent any part of the model's description, input/output format, requirements or code
                    - ❌ DO NOT generate fake code or use your own knowledge about the model
                    - ✅ ONLY use the **actual output** returned from the tool
                    - ✅ Include tool content in your final answer exactly as returned (especially code)

                    ## 🧠 Output Structure (Final Answer)

                    After retrieving tool results, choose best-fit model for the task (You may need to choose more than one model for some tasks), only choose the model 
                    that strictly relevant to the task and only return the output of the model result in the following format:

                    **✅ Model name and link**  
                    `<model name>` — `<link>`

                    **📝 Description (from tool):**  
                    <model description>

                    **📥 Input format (from tool):**  
                    <description of expected input>

                    **📤 Output format (from tool):**  
                    <description of model output>

                    **🛠️ Library Requirements (from tool) **
                    <requirements to use the model>

                    **🧪 Example code (from tool):**  
                    ```python
                    <exact code snippet from tool>

                    You MUSTN'T return any of your thought process, only the model details.

                    Here are the list of models and there corresponding URL:
                    {model_list}
                    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("ai", "{agent_scratchpad}"),
        ("human", "{input}")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    try:
        result = executor.invoke({"input": state['task'], "model_list": state['models'].models})
        state['model_description'] = result['output']
        return state
    except Exception as e:
        print(f"Error occur in model selection tool calling phase: {e}")
        state['error'] = Error.MODEL_TOOL_ERROR
        return state