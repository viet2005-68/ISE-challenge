import os
from ..state.agent_state import AgentState

from constants.constants import BASE_URL, LLAMA_MODEL
from ..model.task import Tasks
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from ..model.model_selection import ModelSelection
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from utils.csv_utils import format_model_list
from ..load_data import df_model

llm = ChatOpenAI( 
    base_url=BASE_URL,
    model=LLAMA_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

model_selection_parser = PydanticOutputParser(pydantic_object=ModelSelection)

def model_selection_agent(state: AgentState):
    system_prompt = """
                    You are a machine learning expert assigned to select the suitable model for a given task.
                    Given:
                    - A user description about the task
                    - A list of available model (names and links)
                    Your job is to choose ALL suitable models provided in the list for user's specific tasks.
                    Return the answer in the format
                    {structured_output}
                    Here are the list of model:
                    {model_list}
                    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
    ]).partial(structured_output=model_selection_parser.get_format_instructions())

    chain = prompt | llm | model_selection_parser

    result = chain.invoke({"input": state['task'], "model_list": format_model_list(df_model)})
    state['models'] = result
    return state