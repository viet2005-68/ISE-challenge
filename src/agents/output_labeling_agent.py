import os
from constants.constants import BASE_URL, LLAMA_MODEL
from ..state.agent_state import AgentState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI( 
    base_url=BASE_URL,
    model=LLAMA_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def output_agent(state: AgentState) -> AgentState:
    prompt = r"""
        Your are a specialist in machine learning. Your task is to identify the absolute classes of the given problem description, following with an output description.
        ### Input:
        - Problem description: {problem_description}
        - Output description: {output_description}

        You must return an array, strictly following these guidelines:
        - Understand the context from the given problem description.
        - Extract the class names from the output description. You **must not** invent new labels or translate the class names. Use them exactly as given in the output description.
        - Create an array containing the classes.

        You must return only the array containing those classes, without any formatting.
    """

    prompt = prompt.format(
        problem_description = state['task'],
        output_description = state["subtasks"].subtask_four
    )
    response = llm.invoke(prompt)
    return {**state, "output_classes": response.content}