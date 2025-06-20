from ..state.agent_state import AgentState
from langgraph.graph import StateGraph, END, START

from ..agents.classify_agent import classify_agent
from ..agents.code_evaluator_agent import code_evaluator_agent
from ..agents.coding_agent import coding_agent
from ..agents.dependencies_agent import dependencies_agent
from ..agents.fallback_model_selection_tool_node import fallback_model_selection_tool_node
from ..agents.model_selection_agent import model_selection_agent
from ..agents.model_selection_using_tool_agent import model_selection_using_tool_agent
from ..agents.output_labeling_agent import output_agent
from ..agents.output_parser_agent import output_parser_agent

def init_agent(state: AgentState):
    state = {
        "messages": [],
        "task": state["task"],
        "data": None,
        "subtasks": None,
        "models": None,
        "model_detailed_list": None,
        "output_classes": None,
        "code": None,
        "dependencies": None,
        "error": None
    }
    return state

graph = StateGraph(AgentState)
graph.add_node("fallback_model_selection", lambda state: state)
graph.add_node("init_node", init_agent)
graph.add_node("formulation_node", classify_agent)
graph.add_node("model_selection_node", model_selection_agent)
graph.add_node("model_selection_using_tool_node", model_selection_using_tool_agent)
graph.add_node("model_output_parser_node", output_parser_agent)
graph.add_node("labeling_node", output_agent)
graph.add_node("coding_node", coding_agent)
graph.add_node("evaluation_node", code_evaluator_agent)
graph.add_node("dependencies_node", dependencies_agent)

graph.add_edge(START, "init_node")
graph.add_edge("init_node", "formulation_node")
graph.add_edge("formulation_node", "model_selection_node")
graph.add_edge("model_selection_node", "model_selection_using_tool_node")
graph.add_edge("model_selection_using_tool_node", "fallback_model_selection")
graph.add_conditional_edges(
    "fallback_model_selection",
    fallback_model_selection_tool_node,
    {
        "continue": "model_output_parser_node",
        "repeat": "model_selection_using_tool_node"
    }
)
graph.add_edge("model_output_parser_node", "labeling_node")
graph.add_edge("labeling_node", "coding_node")
graph.add_edge("coding_node", "evaluation_node")
graph.add_edge("evaluation_node", "dependencies_node")
graph.add_edge("dependencies_node", END)

app = graph.compile()