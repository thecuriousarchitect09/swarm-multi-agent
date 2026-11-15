# Swarm Customer Support with LangGraph and Ollama

A local, peer-to-peer multi-agent demo built with LangGraph and Ollama. There is no supervisor agent. Four specialist agents (triage, billing, tech, and returns) share one conversation, and each one decides when to hand control to a peer by calling a handoff tool.

## How it works

- **Triage** greets the customer, asks a clarifying question if needed, and routes to a specialist.
- **Billing** handles charges, invoices, refunds, and payment methods.
- **Tech** troubleshoots devices, software, and account access.
- **Returns** handles return requests, replacements, and return shipping.

Each agent is a node in a LangGraph `StateGraph`. The active agent is stored in the checkpointed state, so the next user message goes straight to the specialist that is currently talking to the customer. Handoffs are capped per turn to prevent ping-pong loops.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running locally
- A model with tool-calling support (the default is `llama3.1`)

## Setup

1. Pull the model:

   ```bash
   ollama pull llama3.1
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate           # Windows
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure the Ollama server is running. On most installs it starts automatically. You can check with:

   ```bash
   ollama list
   ```

## Running

```bash
python swarm_ollama.py
```

Type a message at the `You:` prompt. Each reply is prefixed with the name of the agent that answered, for example `[billing]`. Type `quit` or `exit` to stop.

Example session:

You: I got charged twice this month.
[billing] I’m sorry about the duplicate charge. Can you tell me the date of the second charge?


## Configuration

Edit the constants at the top of `swarm_ollama.py`:

| Constant | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `"llama3.1"` | Any Ollama model with tool-calling support |
| `MAX_HANDOFFS_PER_TURN` | `4` | Maximum transfers allowed in one user turn |

To use a different model, pull it with `ollama pull <name>` and update `MODEL_NAME`. Smaller models often misroute requests, so test a few sample messages after switching.

## Project structure

.
├── swarm_ollama.py # Graph, agents, handoff tools, and chat loop
├── requirements.txt # Python dependencies
└── README.md # This file


## Extending the swarm

- **Add a new agent:** create a handoff tool with a clear docstring, add an entry to `AGENTS` with its prompt and allowed handoffs, then register it in the graph builder and in `HANDOFF_TARGETS`. Also add a handoff tool to any existing agent that should be able to reach it.
- **Add real tools:** give agents lookups such as `get_invoice(order_id)` and wire in LangGraph's `ToolNode` so they can act on results before replying.
- **Persist conversations:** replace `MemorySaver` with `SqliteSaver` to keep state across restarts.
- **Visualize the topology:** run `graph.get_graph().draw_mermaid()` to print the allowed handoffs as a Mermaid diagram.

## Troubleshooting

- **`connection refused` or similar errors:** Ollama is not running. Start it or run `ollama serve`.
- **`model not found`:** run `ollama pull llama3.1` (or the model named in `MODEL_NAME`).
- **Wrong agent answers:** the model may be too small for reliable tool calling. Try a larger model or refine the handoff tool docstrings.
- **Handoffs stop mid-conversation:** you may have reached `MAX_HANDOFFS_PER_TURN`. Raise it or check the routing prompts for loops.