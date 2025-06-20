import os
from ..state.agent_state import AgentState
from langchain_openai import ChatOpenAI
from constants.constants import BASE_URL, LLAMA_MODEL

llm_coding = ChatOpenAI(
    # api_key=os.getenv("OPENAI_API_KEY"),
    # model=os.getenv("MODEL_NAME"),
    base_url=BASE_URL,
    model='llama-3.3-70b-versatile',
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5
)

def coding_agent(state: AgentState) -> AgentState:
    base_prompt = r"""
    You are a specialist in Machine Learning. Your task is to generate a *fully functional with all necessary imports and dependencies* codebase in *Python* that can be executed flawlessly.

    You will be provided with:
    - A problem description
    - An input specification
    - An output specification
    - A description of models that you needed to use
    - A list of model to use to solve the problem

    ### Input:
    - Problem description: {problem_description}
    - Problem input specification: {problem_input_description}
    - Problem output specification: {problem_output_description}
    - ML Model list and there detailed description: {model_list}
    - Output classes: {output_classes}

    ### Guidelines

    You *must* strictly follow the following guidelines:
    - DO NOT generate any new input. ONLY use the given input file form the problem input description provided above.
    - If NO input file is provided, assume that the input file will be "test.csv"
    - The preprocessing step should be suitable for the data type.
    - The postprocessing step should notices the differences between the data returned by the model and the output requirements. You must extract and use the exact class labels as defined in the output specification.
    - Do *not invent new labels or translate* the class names. Use them exactly as given.
    - You *must* make sure that your codebase can be executed flawlessly that would not encounter any errors or exceptions.
    - You must add some tqdm to see the infer progress.
    - Output file name MUST be "predictions.csv"

    Your implementation *must strictly follow* the structure below:
    1. *Imports*: All required libraries.
    2. *Preprocessing*: Handle and transform the input as defined.
    3. *Inference logic*: Use the described model for prediction. You *must* use tqdm or similar logging library to track progress.
    4. *Postprocessing*: Format or transform the raw output into the final result as described.
    5. *Output*: Export the predict results into a suitable file as describe above (MUST BE CSV FILE)

    You must *not* include any explanations, markdown, or logging outside what is required by the problem.

    Return *only* the complete Python codebase, and you **MUST NOT** include a main function in any way. Wrap it with:
    \`\`\`python
    # code here
    \`\`\`
    """

    prompt = base_prompt.format(
        problem_description=state["task"],
        problem_input_description=state['subtasks'].subtask_three,
        problem_output_description=state['subtasks'].subtask_four,
        model_description=state["model_description"],
        model_list=state['model_detailed_list'].models,
        output_classes=state["output_classes"],
    )
    response = llm_coding.invoke(prompt)
    state['code'] = response.content
    # return {**state, "code": response.content}
    return state