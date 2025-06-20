import os
from ..model.model_list import ModelList
from langchain_core.output_parsers import PydanticOutputParser
from constants.constants import BASE_URL, LLAMA_MODEL
from ..state.agent_state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


parser = PydanticOutputParser(pydantic_object=ModelList)

output_parser_llm = ChatOpenAI(
    base_url=BASE_URL,
    model=LLAMA_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def recovery_parser_agent(state: AgentState):
    recovery_prompt = """
        The previous attempt to extract structured data failed.
        You must now attempt to salvage **as much accurate information as possible** from the input text.

        Try to extract the following fields. It's okay to leave some fields blank if information isn't available:
        - `model_description`
        - `model_input_format`
        - `model_output_format`
        - `model_requirements`
        - `model_sample_code`

        Do NOT make up any data. Only extract what you are certain is present in the input.

        Format your output as JSON:
        {formatted_output}
    """
    prompt = ChatPromptTemplate.from_messages([
        ('system', recovery_prompt),
        ('human', "{input}")
    ]).partial(formatted_output=parser.get_format_instructions())

    chain = prompt | output_parser_llm | parser
    try:
        response = chain.invoke({"input": state['model_description']})
        state['model_detailed_list'] = response
        state['recovery_used'] = True
    except Exception as e:
        print(f"❌ Recovery also failed: {e}")
        state['model_detailed_list'] = {
            "error": "Both primary and recovery parsing failed.",
            "details": str(e),
            "raw_input": state.get('model_description', '')
        }
        state['recovery_used'] = True
    return state


def output_parser_agent(state: AgentState):
    system_prompt = """
                    You are a smart AI tasked with extracting structured technical details about a machine learning model from a reasoning result.
                    You are given a detailed text description about a model (or a list of model).
                    Your goal is to fill the following fields using the data about chosen models based on the text:

                    - `model_description`: A detailed explanation of what the model is and what it does.
                    - `model_input_format`: A detailed description of the model's input format, including dimensions, data types, and expected preprocessing if mentioned.
                    - `model_output_format`: A detailed description of the output format including dimensions, data types, label name and its meaning.
                    - `model_requirements`: A detailed description about the requirements needed to be sastified in order to use the model
                    - `model_sample_code`: A sample code on how to use the model.

                    You MUSTN'T create any data on your own, only using the data provided in the text.

                    Return the data as a JSON object matching the following structure:
                    {formatted_output}
                    """
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "{input}")
    ]).partial(formatted_output=parser.get_format_instructions())
    chain = prompt | output_parser_llm | parser
    try:
        response = chain.invoke({"input": state['model_description']})
        state['model_detailed_list'] = response
    except Exception as e:
        print(f"⚠️ Output parsing failed: {e}")
        # Optional: fallback, recovery, or debug output
        state['model_detailed_list'] = {
            "error": "Failed to parse output",
            "details": str(e),
            "raw_input": state.get('model_description', '')
        }
        state = recovery_parser_agent(state)
    return state