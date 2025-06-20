from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from ..model.dependencies import Dependencies
from ..model.model_list import ModelList
from ..model.model_selection import ModelSelection
from ..model.task import Tasks

class AgentState(TypedDict):
    task: HumanMessage | None
    data: str | None
    messages: Annotated[List[BaseMessage], add_messages]
    subtasks: Tasks
    models: ModelSelection
    model_description: str
    model_detailed_list: ModelList
    output_classes: str | None
    code: str | None
    improved_code: str | None
    evaluation_summary: str | None
    dependencies: Dependencies | None
    error: str | None