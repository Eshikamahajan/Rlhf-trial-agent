import json
import os
import random
from typing import TypedDict, List, Optional

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


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return _llm

# ---------------------------------------------------------------------------
# 0. DOMAIN NOTE
# ---------------------------------------------------------------------------
# This agent handles delivery-app support tickets: order delays, app/system
# glitches, payment hiccups, and backend/API issues (e.g. an OpenAI API call
# timing out inside the app's own chat support feature).
#
# Severity is intentionally modeled on a 3-step scale (minor / moderate /
# major) with realistic support-desk thresholds. A short delay or a one-off
# glitch should land as "minor" and get an automatic, low-friction resolution
# (a notification + a small coupon) rather than triggering an escalation.
# Only genuinely disruptive situations (long delays, repeated failures,
# backend outages) should reach "major" and go to a human agent. This keeps
# the demo realistic -- not every hiccup is treated like a crisis.

# ---------------------------------------------------------------------------
# 1. RLHF setup: Q-table
# ---------------------------------------------------------------------------
Q_TABLE_PATH = "delivery_q_table.json"
ACTIONS = [
    "ask_more_questions",
    "check_status_tool",
    "offer_resolution_tool",
    "escalate_to_support",
]
ALPHA = 0.3     # learning rate
GAMMA = 0.8     # discount factor
EPSILON = 0.2   # exploration rate (only used once a state has learned data)
MAX_TURNS = 3   # keep episodes short for the demo


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


def rule_based_action(context: dict) -> str:
    """
    Instruction-based fallback used ONLY when the current state has never
    been seen before (i.e. there's no learned Q-values to exploit yet).

    Instead of picking a random action to "explore" a state we know nothing
    about, we use a small set of support-desk rules so the agent's first
    encounter with a new situation is still a sensible, explainable decision.
    Once this state has been visited and has real Q-values, normal
    epsilon-greedy exploration/exploitation takes over in choose_action().
    """
    issue_type = context.get("issue_type", "unknown")
    delay_minutes = context.get("delay_minutes", 0)
    severity = context.get("severity", "unknown")
    turn = context.get("turn", 0)

    # We haven't even assessed the situation yet -> assess it first.
    if severity == "unknown":
        return "check_status_tool"

    # Backend-side problems (system outage / API issues) aren't something
    # the customer can wait out, so escalate a bit sooner than user-side
    # delays once they're past "minor".
    if issue_type in {"system_outage", "api_issue"} and severity in {"moderate", "major"}:
        return "escalate_to_support"

    if severity == "major":
        return "escalate_to_support"

    if severity == "moderate":
        return "offer_resolution_tool"

    if severity == "minor":
        # Early in the conversation, a quick clarifying question is cheap
        # and can avoid over- or under-resolving. Later, just close it out.
        if turn == 0:
            return "ask_more_questions"
        return "offer_resolution_tool"

    return "ask_more_questions"


def choose_action(q: dict, state: str, context: Optional[dict] = None) -> str:
    """
    - If this state has learned Q-values already, use epsilon-greedy
      (explore/exploit) as normal.
    - If this state has NEVER been seen before, don't explore randomly.
      Use rule_based_action() to make an instruction-driven first decision,
      and initialize its Q-values so future visits can learn from it.
    """
    if state not in q:
        get_q_values(q, state)  # initialize zeros for future learning
        return rule_based_action(context or {})

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
def check_status_tool(issue_type: str, delay_minutes: int, user_type: str) -> str:
    """
    Tool 1: classifies severity as minor / moderate / major.
    Falls back to simple, calibrated rules if no API key is set.
    Thresholds are deliberately modest: a short delay is "minor", not a crisis.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        if issue_type in {"system_outage", "api_issue"}:
            return "major" if delay_minutes > 60 else "moderate"
        if delay_minutes <= 15:
            return "minor"
        elif delay_minutes <= 45:
            return "moderate"
        else:
            return "major"

    prompt = (
        f"A delivery-app customer support ticket has issue_type='{issue_type}', "
        f"delay_minutes={delay_minutes}, user_type='{user_type}'. "
        "This is routine customer support triage, not an emergency. "
        "Classify severity as exactly one word: minor, moderate, or major "
        "(minor = small delay/glitch, moderate = noticeable inconvenience, "
        "major = repeated failure or backend outage affecting the order). "
        "Answer with only that one word."
    )
    reply = get_llm().invoke(prompt).content.strip().lower()
    return reply if reply in {"minor", "moderate", "major"} else "moderate"


def offer_resolution_tool(severity: str) -> str:
    """Tool 2: maps severity to a recommended next step for the ticket."""
    mapping = {
        "minor": "Send an apology notification + a small delay-compensation coupon. No escalation needed.",
        "moderate": "Offer a refund/reschedule option and bump the order's priority.",
        "major": "Escalate to a human support agent / on-call engineer and share an ETA with the customer.",
    }
    return mapping.get(severity, "Ask more questions to clarify the ticket.")


# ---------------------------------------------------------------------------
# 3. LANGGRAPH STATE
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    issue_signals: List[str]   # e.g. ["order_delay_20min", "confirmed_still_in_transit"]
    issue_type: str            # order_delay | wrong_item | payment_issue | system_outage | api_issue
    delay_minutes: int
    user_type: str             # new | regular | premium
    severity: str              # unknown | minor | moderate | major
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
    # state_key = a simple discrete signature of the ticket so far
    state["state_key"] = (
        f"issue={state['issue_type']}_severity={state['severity']}_user={state['user_type']}"
    )
    return state


def choose_action_node(state: AgentState) -> AgentState:
    q = state["q_table"]
    context = {
        "issue_type": state["issue_type"],
        "delay_minutes": state["delay_minutes"],
        "severity": state["severity"],
        "turn": state["turn"],
    }
    action = choose_action(q, state["state_key"], context)
    state["action"] = action
    print(f"\n[Turn {state['turn']}] State: {state['state_key']} -> Agent picks action: {action}")
    return state


def act_node(state: AgentState) -> AgentState:
    action = state["action"]

    if action == "check_status_tool":
        state["severity"] = check_status_tool(
            state["issue_type"], state["delay_minutes"], state["user_type"]
        )
        state["output"] = f"Tool 1 result -> severity classified as: {state['severity']}"

    elif action == "offer_resolution_tool":
        state["output"] = f"Tool 2 result -> {offer_resolution_tool(state['severity'])}"

    elif action == "ask_more_questions":
        # simulate collecting one more signal (in a real app, ask the user)
        extra_signal, extra_delay = random.choice(
            [
                ("restaurant_confirmed_prep_delay", 5),
                ("rider_stuck_in_traffic", 10),
                ("payment_gateway_slow_response", 3),
                ("app_reported_minor_glitch", 0),
            ]
        )
        state["issue_signals"].append(extra_signal)
        state["delay_minutes"] += extra_delay
        state["output"] = f"Agent asked a follow-up question. New signal noted: {extra_signal}"

    elif action == "escalate_to_support":
        state["output"] = "Agent escalated the ticket directly to a human support agent."

    print(f"  -> {state['output']}")
    return state


def support_feedback_node(state: AgentState) -> AgentState:
    """This is the RLHF step: a human (support lead) rates the agent's last action."""
    print("  Rate this action: (1 = good, 0 = neutral, -1 = bad/unhelpful)")
    try:
        reward = float(input("  Support lead feedback: ").strip())
    except (ValueError, EOFError):
        reward = 0.0  # default if no input (e.g. automated run)
    state["reward"] = reward
    return state


def update_q_node(state: AgentState) -> AgentState:
    q = state["q_table"]
    next_state_key = (
        f"issue={state['issue_type']}_severity={state['severity']}_user={state['user_type']}"
    )
    update_q_table(q, state["state_key"], state["action"], state["reward"], next_state_key)
    state["turn"] += 1
    return state


def should_continue(state: AgentState) -> str:
    if state["turn"] >= MAX_TURNS or state["action"] == "escalate_to_support":
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
    graph.add_node("support_feedback", support_feedback_node)
    graph.add_node("update_q", update_q_node)

    graph.set_entry_point("classify_state")
    graph.add_edge("classify_state", "choose_action")
    graph.add_edge("choose_action", "act")
    graph.add_edge("act", "support_feedback")
    graph.add_edge("support_feedback", "update_q")
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
        "issue_signals": ["order_running_late"],
        "issue_type": "order_delay",   # try: "wrong_item", "payment_issue", "system_outage", "api_issue"
        "delay_minutes": 12,
        "user_type": "regular",        # new | regular | premium
        "severity": "unknown",
        "state_key": "",
        "action": "",
        "output": "",
        "turn": 0,
        "q_table": q_table,
        "reward": 0.0,
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n--- Ticket resolved ---")
    print("Final Q-table (saved to delivery_q_table.json):")
    print(json.dumps(final_state["q_table"], indent=2))
