import os
from ..state.agent_state import AgentState
from langchain_openai import ChatOpenAI

llm_code_evaluator = ChatOpenAI(
    model= 'gpt-4.1-nano-2025-04-14',
    api_key= os.getenv('OPENAI_API_KEY'),
    temperature=0.25
)

def code_evaluator_agent(state: AgentState) -> AgentState:
    system_prompt = """
    You are an expert code reviewer and machine learning engineer. Your task is to evaluate and improve the given Python code.
    
    You will be provided with:
    - The original code
    - Problem description and requirements
    - Model information
    
    ### Input:
    - Code to evaluate: {code}
    - Problem description: {problem_description}
    - Problem input specification: {problem_input_description}
    - Problem output specification: {problem_output_description}
    - ML model description: {model_description}
    - List of ML model and there details: {model_list}
    - Output classes: {output_classes}

    ### Evaluation Criteria
    - MANDATORY: Ensure input data is preprocess before it go to model
    - MANDATORY: Ensure input data shape of the model in {model_description} exactly matches model's expected input format
    - MANDATORY: Validate tensor dimensions and batch sizes before model prediction
    - MANDATORY: Handle shape mismatches with proper reshaping/preprocessing
    - MANDATORY: Add explicit shape validation and error handling for model I/O
    - MANDATORY: ENSURE models input format MATCHES: {model_input_form}
    - MANDATORY: Ensure model output matches: {output_classes}
    - MANDATORY: Add shape debugging information when shape errors occur

    ### Output Format:
    You must return:
    1. An evaluation summary highlighting strengths and areas for improvement
    2. The improved version of the code incorporating all necessary enhancements
    3. You MUST NOT contain logging code
    
    ### STRICT RULES:
    - Ensure input data shape exactly matches model's expected input format
    - You MUST NOT use any other models beside the ones defined in the provided model list.
    - You MUST ONLY modify how the model works, including necessary processing steps and the model behaviour. DO NOT change the model architecture or any other code fields.
    - DO NOT generate any new input. Only use the provided input files from the original codebase.
    - DO NOT include any main() function or code block (no if _name_ == "_main_" or similar).
    - DO NOT include any logging code
    - DO NOT change how the output is generated — the output must remain identical in structure and content.
    - ONLY MODIFY THE ORGINAL CODE BASE IF IT IS NECESSARY TO RUNTIME ERRORS
    
    ### EXECUTION REQUIREMENTS
    - Code must be immediately executable without any user intervention
    - Code must export a file of predicted labels in the same format and behaviour in the previous codebase.
    - Results should be generated automatically when the script is run
    
    ### The improved code must:
    - DO NOT change how the output is generated — the output must remain identical in structure and content.
    - Include all necessary imports
    - Only modify the model behaviour and input shapes if being mismatch
    - Add input validation (especially shape validation for ML models)
    - Add proper tensor reshaping to match model's expected input format
    - Handle batch dimensions correctly
    - Add clear error messages for shape mismatches
    - Wrap it with:
    ```python
    # code here
    ```
    """
    prompt = system_prompt.format(
        code=state["code"],
        problem_description=state["task"],
        problem_input_description=state['subtasks'].subtask_three,
        problem_output_description=state['subtasks'].subtask_four,
        model_description=state["model_description"],
        model_list=state['model_detailed_list'],
        model_input_form=[{x.model_description, x.model_input_format} for x in state['model_detailed_list'].models],
        output_classes=state['output_classes']
    )

    response = llm_code_evaluator.invoke(prompt)

    # Extract evaluation summary and improved code from response
    content = response.content
    parts = content.split("```python")
    print(content)
    if len(parts) > 1:
        evaluation_summary = parts[0].strip()
        improved_code = parts[1].split("```")[0].strip()
    else:
        evaluation_summary = content
        improved_code = state["code"]

    return {
        **state,
        "evaluation_summary": evaluation_summary,
        "code": improved_code,
    }
