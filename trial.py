import json
import os
import random
from typing import TypedDict, List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()

_llm = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

else:
    print("GOOGLE_API_KEY loaded successfully.")

from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return _llm

# setting up space for rlhf
Q_TABLE_PATH = "q_table.json"
ACTIONS = ["ask_more_questions", "call_urgency_tool", "call_care_tool", "refer_doctor"]
ALPHA = 0.3     # learning rate
GAMMA = 0.8     # discount factor
EPSILON = 0.2   # exploration rate
MAX_TURNS = 3   # keep episodes short for the demo

# ---------------------------------------------------------------------------
# 1. Q-TABLE: load / save / update
# ---------------------------------------------------------------------------
def load_q_table() -> dict:
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_q_table(q: dict) -> None:
    with open(Q_TABLE_PATH, "w") as f:
        json.dump(q, f, indent=2)


def get_q_values(q: dict, state: str) -> dict:
    return q.setdefault(state, {a: 0.0 for a in ACTIONS})


# def choose_action(q: dict, state: str) -> str:
#     """Epsilon-greedy action selection."""
#     if random.random() < EPSILON:
#         return random.choice(ACTIONS)
#     q_values = get_q_values(q, state)
#     return max(q_values, key=q_values.get)

def choose_action(q: dict, state: str) -> str:
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_values = get_q_values(q, state)
    best = max(q_values.values())
    return random.choice([a for a, v in q_values.items() if v == best])


def update_q_table(q: dict, state: str, action: str, reward: float, next_state: str) -> None:
    current = get_q_values(q, state)
    next_max = max(get_q_values(q, next_state).values(), default=0.0)
    current[action] += ALPHA * (reward + GAMMA * next_max - current[action])
    save_q_table(q)


# ---------------------------------------------------------------------------
# 2. THE ONLY 2 TOOLS
# ---------------------------------------------------------------------------
def check_urgency_tool(symptoms: List[str]) -> str:
    """Tool 1: Gemini classifies urgency (falls back to rules if no API key)."""
    if not os.environ.get("GOOGLE_API_KEY"):
        high_risk = {"chest pain", "difficulty breathing", "severe bleeding"}
        if any(s in high_risk for s in symptoms):
            return "high"
        return "medium" if len(symptoms) >= 2 else "low"

    prompt = (
        f"Patient symptoms: {', '.join(symptoms)}. "
        "Classify urgency as exactly one word: high, medium, or low. "
        "Answer with only that one word."
    )
    reply = get_llm().invoke(prompt).content.strip().lower()
    return reply if reply in {"high", "medium", "low"} else "medium"


def suggest_care_tool(urgency: str) -> str:
    """Tool 2: maps urgency to a recommended next step."""
    mapping = {
        "high": "Escalate immediately to emergency care.",
        "medium": "Recommend a same-week doctor visit.",
        "low": "Suggest home care / OTC guidance, monitor symptoms.",
    }
    return mapping.get(urgency, "Ask more questions to clarify.")


# ---------------------------------------------------------------------------
# 3. LANGGRAPH STATE
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    symptoms: List[str]
    urgency: str
    state_key: str
    action: str
    output: str
    turn: int
    q_table: dict
    reward: float

# ---------------------------------------------------------------------------
# 4. NODES
# ---------------------------------------------------------------------------
def classify_state_node(state: AgentState) -> AgentState:
    # state_key = a simple discrete signature of the conversation so far
    state["state_key"] = f"symptoms={len(state['symptoms'])}_urgency={state['urgency']}"
    return state


def choose_action_node(state: AgentState) -> AgentState:
    q = state["q_table"]
    action = choose_action(q, state["state_key"])
    state["action"] = action
    print(f"\n[Turn {state['turn']}] State: {state['state_key']} -> Agent picks action: {action}")
    return state


def act_node(state: AgentState) -> AgentState:
    action = state["action"]

    if action == "call_urgency_tool":
        state["urgency"] = check_urgency_tool(state["symptoms"])
        state["output"] = f"Tool 1 result -> urgency classified as: {state['urgency']}"

    elif action == "call_care_tool":
        state["output"] = f"Tool 2 result -> {suggest_care_tool(state['urgency'])}"

    elif action == "ask_more_questions":
        # simulate collecting one more symptom (in a real app, ask the user)
        extra = random.choice(["fatigue", "mild headache", "chest pain"])
        state["symptoms"].append(extra)
        state["output"] = f"Agent asked a follow-up question. New symptom noted: {extra}"

    elif action == "refer_doctor":
        state["output"] = "Agent directly referred the patient to a doctor."

    print(f"  -> {state['output']}")
    return state


def human_feedback_node(state: AgentState) -> AgentState:
    """This is the RLHF step: a human rates the agent's last action."""
    print("  Rate this action: (1 = good, 0 = neutral, -1 = bad/unsafe)")
    try:
        reward = float(input("  Clinician feedback: ").strip())
    except (ValueError, EOFError):
        reward = 0.0  # default if no input (e.g. automated run)
    state["reward"] = reward
    return state


def update_q_node(state: AgentState) -> AgentState:
    q = state["q_table"]
    next_state_key = f"symptoms={len(state['symptoms'])}_urgency={state['urgency']}"
    update_q_table(q, state["state_key"], state["action"], state["reward"], next_state_key)
    state["turn"] += 1
    return state


def should_continue(state: AgentState) -> str:
    if state["turn"] >= MAX_TURNS or state["action"] == "refer_doctor":
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# 5. BUILD GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_state", classify_state_node)
    graph.add_node("choose_action", choose_action_node)
    graph.add_node("act", act_node)
    graph.add_node("human_feedback", human_feedback_node)
    graph.add_node("update_q", update_q_node)

    graph.set_entry_point("classify_state")
    graph.add_edge("classify_state", "choose_action")
    graph.add_edge("choose_action", "act")
    graph.add_edge("act", "human_feedback")
    graph.add_edge("human_feedback", "update_q")
    graph.add_conditional_edges(
        "update_q",
        should_continue,
        {"continue": "classify_state", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# 6. RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    q_table = load_q_table()

    initial_state: AgentState = {
        "symptoms": ["fever", "cough"],
        "urgency": "unknown",
        "state_key": "",
        "action": "",
        "output": "",
        "turn": 0,
        "q_table": q_table,
        "reward": 0.0,
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n--- Episode finished ---")
    print("Final Q-table (saved to q_table.json):")
    print(json.dumps(final_state["q_table"], indent=2))