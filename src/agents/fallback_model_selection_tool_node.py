from ..model.error import Error
from ..state.agent_state import AgentState

def fallback_model_selection_tool_node(state: AgentState):
    if state['error'] == Error.MODEL_TOOL_ERROR:
        return "repeat"
    return "continue"