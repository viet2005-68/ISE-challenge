import os
from ..state.agent_state import AgentState

from constants.constants import BASE_URL, LLAMA_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from ..model.dependencies import Dependencies
from langchain_core.prompts import ChatPromptTemplate

dependencies_parser = PydanticOutputParser(pydantic_object=Dependencies)

llm = ChatOpenAI( 
    base_url=BASE_URL,
    model=LLAMA_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def dependencies_agent(state: AgentState) -> AgentState:
    system_prompt="""
                You are a code analysis assistant. Given the contents of a Python file, analyze all import and from ... import ... statements to extract the full list of dependencies.
                Your goal is to return a clean list of external modules and packages that would need to be installed via pip in order to run the code.
                Ignore built-in Python modules (like os, sys, math, datetime, etc.).
                Include third-party libraries such as numpy, pandas, requests, etc.
                If a module is imported using an alias (e.g., import numpy as np), resolve it to its base name (numpy).
                Only output the unique package names (not the specific submodules).
                Do not include relative or local imports (e.g., from .utils import helper).
                Use the following known mappings when resolving packages:

                - `PIL` → `pillow`
                - `cv2` → `opencv-python`
                - `sklearn` → `scikit-learn`
                - `yaml` → `pyyaml`
                - `Crypto` → `pycryptodome`
                - `bs4` → `beautifulsoup4`
                - `tensorflow.keras` → `tensorflow`
                - `email` and `html` (standard lib) → DO NOT install
                Input:
                # Python code goes here  
                {code}
                
                Output:
                Make sure return the answer in the following format:
                {formatted_output}
                  """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ]).partial(formatted_output=dependencies_parser.get_format_instructions())

    chain = prompt | llm | dependencies_parser

    result = chain.invoke({"code": state['code']})
    state['dependencies'] = result.dependencies
    return state