from typing import TypedDict, Literal
from google.genai import Client, types
import argparse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Client()


class ResearchState(TypedDict):
    # User input (immutable)
    question: str
    max_steps: int

    # Control
    action: str
    step_count: int
    reasoning: str

    # Context
    observations: list[str]
    visited_topics: set[str]

    # Output
    final_answer: str


def search_knowledge_base(topic: str) -> str:
    knowledge_base = {
        "pricing": (
            "Start plan: $9/month, up to 5 users. "
            "Pro plan: $49/month, up to 25 users, includes API access."
            "Enterpise: $199/month, unlimited users, dedicated support."
        ),
        "policies": (
            "Refund policy: full refund within 30 days. "
            "Cancellation: cancel anytime, no penalties. "
            "Data retention: data deleted 30 days after account closure."
        ),
        "technical": (
            "API rate limits:: 1000 req/day on Pro, 10 000 req/day on Enterprise. "
            "Supported langugages: Python, Java, Go. "
            "Uptime SLA: 99.9% for Enterpise, best-effort for Starter and Pro."
        ),
    }
    return knowledge_base.get(topic, "No information found for that topic.")


def planner(state: ResearchState) -> dict:
    observations_text = ""
    if state["observations"]:
        observations_text = "Observations gathered so far:\n"
        for i, obs in enumerate(state["observations"]):
            observations_text += f" {i}. {obs}\n"

    search_text = ""
    if state["visited_topics"]:
        search_text = (
            "Topic already visited (do not search these again): "
            f"{', '.join(state['visited_topics'])}\n\n"
        )

    force_done = state["step_count"] >= state["max_steps"]
    prompt = (
        "You are a research planner for a support assistant.\n\n"
        f"Question: {state['question']}\n\n"
        f"{observations_text}\n"
        f"{search_text}"
        f"Steps used so far: {state['step_count']} of {state['max_steps']} allowed.\n\n"
        "Decide what to do next. Response with exactly one of:\n"
        " search_pricing   - to look up plan and pricing information\n"
        " search_policies  - to look up refund, cancellation and data policies\n"
        " search_technical - to look up API, language and uptime information\n"
        " done             - if the observations already contain enough to answer\n"
        "Then on the next line write one sentence explaining your reasoning.\n"
        "Format:\n"
        "ACTION: <one of the four options above>\n"
        "REASON: <one sentence>\n"
    )

    if force_done:
        return {
            "action": "done",
            "reasoning": f"Max steps ({state['max_steps']}) reached. Proceeding to synthesis.",
        }
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )
    lines = response.text.strip().splitlines()

    action = "done"
    reasoning = ""
    for line in lines:
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("REASON:"):
            reasoning = line.split(":", 1)[1].strip()

    valid_actions = {"search_pricing", "search_policies", "search_technical", "done"}
    if action not in valid_actions:
        action = "done"
        reasoning = f"Unrecognised action '{action}'. Defaulting to done."

    return {
        "action": action,
        "reasoning": reasoning,
        "step_count": state["step_count"] + 1,
    }


def executor(state: ResearchState) -> dict:
    topic_map = {
        "search_policies": "policies",
        "search_technical": "technical",
        "search_pricing": "pricing",
    }
    topic = topic_map.get(state["action"], "")
    result = search_knowledge_base(topic)
    label = state["action"].replace("search_", "").title()
    observation = f"[{label}] {result}"

    state["observations"].append(observation)
    state["visited_topics"].add(topic)
    return {
        "observations": state["observations"],
        "visited_topics": state["visited_topics"],
    }


def synthesize(state: ResearchState) -> dict:
    observations_text = "\n".join(state["observations"])
    prompt = (
        "You are a helpful support assistant.\n"
        "Answer the question using only the information below. Be concise (3-5 senetences).\n\n"
        f"Question: {state['question']}\n\n"
        f"Information gathered:\n{observations_text}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )
    return {"final_answer": response.text}


def route_planner(state: ResearchState) -> Literal["executor", "synthesize"]:
    if state["action"] == "done":
        return "synthesize"
    return "executor"


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_planner)
    builder.add_edge("executor", "planner")
    builder.add_edge("synthesize", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    parser = argparse.ArgumentParser(description="Reasearch Assistant")
    parser.add_argument("--question", type=str, help="Your question")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to produce an answer",
        default=3,
    )

    args = parser.parse_args()

    result = app.invoke(
        {
            "question": args.question,
            "max_steps": args.max_steps,
            "action": "",
            "step_count": 0,
            "reasoning": "",
            "observations": [],
            "final_answer": "",
            "visited_topics": {},
        }
    )
    print(result)
