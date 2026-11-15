"""
Swarm (handoff) pattern with LangGraph + Ollama.

There is no supervisor. Each agent is a node that can hand the conversation
to a peer by calling a handoff tool. The active agent persists across turns,
so the specialist keeps talking to the customer until it hands off again.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MODEL_NAME = "llama3.1"
MAX_HANDOFFS_PER_TURN = 4  # guard against ping-pong loops

llm = ChatOllama(model=MODEL_NAME, temperature=0)


# ---------------------------------------------------------------------------
# 1. Handoff tools. The model "calls" these to transfer control. The docstring
#    is what the model reads to decide when to use each one, so write it
#    carefully.
# ---------------------------------------------------------------------------

@tool
def transfer_to_billing(reason: str) -> str:
    """Transfer the customer to the billing agent. Use for charges, invoices,
    refunds, duplicate payments, or payment methods. reason: one-sentence summary."""
    return "transfer requested"


@tool
def transfer_to_tech(reason: str) -> str:
    """Transfer the customer to the technical support agent. Use for devices,
    software, login problems, or errors. reason: one-sentence summary."""
    return "transfer requested"


@tool
def transfer_to_returns(reason: str) -> str:
    """Transfer the customer to the returns agent. Use for return requests,
    replacements, and shipping of returned items. reason: one-sentence summary."""
    return "transfer requested"


# Map each tool name to the agent it transfers to.
HANDOFF_TARGETS = {
    "transfer_to_billing": "billing",
    "transfer_to_tech": "tech",
    "transfer_to_returns": "returns",
}
HANDOFF_TOOLS_BY_NAME = {
    "transfer_to_billing": transfer_to_billing,
    "transfer_to_tech": transfer_to_tech,
    "transfer_to_returns": transfer_to_returns,
}


# ---------------------------------------------------------------------------
# 2. Agent definitions. Each agent has a role prompt and the set of peers it
#    may hand off to. An agent never hands off to itself.
# ---------------------------------------------------------------------------

AGENTS = {
    "triage": {
        "prompt": (
            "You are the triage agent for a customer support company. Your job is "
            "to understand the customer's need and route them. If the request clearly "
            "matches billing, technical support, or returns, call the matching transfer "
            "tool right away with a one-sentence reason. If the request is unclear, ask "
            "one short clarifying question. Do not answer specialist questions yourself."
        ),
        "handoffs": ["transfer_to_billing", "transfer_to_tech", "transfer_to_returns"],
    },
    "billing": {
        "prompt": (
            "You are the billing agent. You handle charges, invoices, refunds, and "
            "payment methods. Help the customer directly when the issue is billing. "
            "If the real problem is a device or software fault, call transfer_to_tech. "
            "If it is about sending back or replacing an item, call transfer_to_returns. "
            "Speak to the customer in your own voice and do not ask them to repeat "
            "anything already in the conversation."
        ),
        "handoffs": ["transfer_to_tech", "transfer_to_returns"],
    },
    "tech": {
        "prompt": (
            "You are the technical support agent. You troubleshoot devices, software, "
            "and account access. Give clear, numbered troubleshooting steps. If the "
            "problem is really a charge or refund, call transfer_to_billing. If the "
            "device is beyond repair and needs replacing, call transfer_to_returns."
        ),
        "handoffs": ["transfer_to_billing", "transfer_to_returns"],
    },
    "returns": {
        "prompt": (
            "You are the returns agent. You handle return requests, replacements, and "
            "return shipping. Confirm the item and reason, then explain the next steps. "
            "If the customer needs a refund or charge correction, call transfer_to_billing. "
            "If the item is still malfunctioning and needs troubleshooting first, call "
            "transfer_to_tech."
        ),
        "handoffs": ["transfer_to_billing", "transfer_to_tech"],
    },
}


# ---------------------------------------------------------------------------
# 3. Graph state
# ---------------------------------------------------------------------------

class SwarmState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str        # persists across turns: who is talking to the user
    next_agent: str          # transient: set only while a handoff is pending
    handoffs_this_turn: int  # budget counter, reset at the start of each turn


# ---------------------------------------------------------------------------
# 4. Graph nodes and routing
# ---------------------------------------------------------------------------

def reset_turn(state: SwarmState) -> dict:
    """Runs at the start of every user turn to clear transient fields."""
    return {"next_agent": "", "handoffs_this_turn": 0}


def entry_router(state: SwarmState) -> str:
    """Send the new user message to whichever agent is currently active."""
    return state.get("active_agent") or "triage"


def make_agent_node(name: str):
    """Build a graph node for one agent in the swarm."""
    spec = AGENTS[name]
    allowed_tools = [HANDOFF_TOOLS_BY_NAME[t] for t in spec["handoffs"]]

    def agent_node(state: SwarmState) -> dict:
        budget_left = state["handoffs_this_turn"] < MAX_HANDOFFS_PER_TURN

        # Once the handoff budget is spent, the agent gets no transfer tools,
        # so it must answer the customer directly.
        model = llm.bind_tools(allowed_tools) if budget_left else llm

        messages = [SystemMessage(content=spec["prompt"]), *state["messages"]]
        ai_msg = model.invoke(messages)

        # Every tool call needs a matching ToolMessage, or the message history
        # becomes invalid for the next model call.
        tool_msgs: list[ToolMessage] = []
        target = None
        for call in ai_msg.tool_calls:
            target_for_call = HANDOFF_TARGETS.get(call["name"])
            if target_for_call and target is None:
                target = target_for_call
                reason = call["args"].get("reason", "")
                content = f"Handed off to the {target} agent. Reason: {reason}"
            else:
                content = "Ignored: only one handoff is allowed per response."
            tool_msgs.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if target is None:
            # No handoff: this agent answers the customer and the turn ends.
            return {
                "messages": [ai_msg],
                "active_agent": name,
                "next_agent": "",
            }

        # Handoff: the target agent becomes active and runs next. The handoff
        # reason travels in the ToolMessage, so the new agent has the context.
        return {
            "messages": [ai_msg, *tool_msgs],
            "active_agent": target,
            "next_agent": target,
            "handoffs_this_turn": state["handoffs_this_turn"] + 1,
        }

    return agent_node


def after_agent(state: SwarmState) -> str:
    """If a handoff is pending, run the target agent. Otherwise end the turn."""
    return state.get("next_agent") or END


# ---------------------------------------------------------------------------
# 5. Assemble and compile the graph
# ---------------------------------------------------------------------------

builder = StateGraph(SwarmState)
builder.add_node("reset_turn", reset_turn)
for agent_name in AGENTS:
    builder.add_node(agent_name, make_agent_node(agent_name))

builder.add_edge(START, "reset_turn")
builder.add_conditional_edges("reset_turn", entry_router, list(AGENTS))

for agent_name in AGENTS:
    builder.add_conditional_edges(
        agent_name,
        after_agent,
        {**{n: n for n in AGENTS}, END: END},
    )

# MemorySaver keeps conversation state per thread_id between turns.
graph = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 6. Simple terminal chat loop
# ---------------------------------------------------------------------------

def main() -> None:
    config = {"configurable": {"thread_id": "customer-session-1"}}
    print("Swarm support demo (type 'quit' to exit)\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config,
        )

        reply = next(
            (m.content for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and m.content),
            "",
        )
        print(f"[{result['active_agent']}] {reply}\n")


if __name__ == "__main__":
    main()