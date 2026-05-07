from typing import TypedDict, Literal
from groq import Groq
import argparse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Groq()


class SupportState(TypedDict):
    user_input: str
    route: str
    response_text: str


def classify_request(state: SupportState) -> dict:
    text = state["user_input"].lower()
    if "refund" in text or "legal" in text:
        return {"route": "approval"}
    elif "unclear" in text or "maybe" in text:
        return {"route": "clarify"}
    elif "invoice" in text or "charge" in text:
        return {"route": "billing"}
    else:
        return {"route": "answer"}


def billing_helper(state: SupportState) -> dict:
    return {
        "response_text": (
            "To look into this, please share your invoice ID "
            "and the approximate date of the charge."
        )
    }


def ask_followup(state: SupportState) -> dict:
    return {
        "response_text": (
            "I want to make sure I help you accurately. "
            "Could you give me one more detail about what you need?"
        )
    }


def approval_gate(state: SupportState) -> dict:
    return {
        "response_text": (
            "Your request involves a topic that requires a team member to review. "
            "We will follow up with you shortly."
        )
    }


def draft_answer(state: SupportState) -> dict:
    prompt = (
        "You are a helpful customer support assistant.\n"
        "Answer the following question clearly and in 3-5 lines.\n\n"
        f"Customer question: {state['user_input']}"
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
    )
    return {"response_text": response.choices[0].message.content}


def choose_next_step(
    state: SupportState,
) -> Literal["ask_followup", "draft_answer", "approval_gate", "billing_helper"]:
    route_map = {
        "clarify": "ask_followup",
        "approval": "approval_gate",
        "answer": "draft_answer",
        "billing": "billing_helper",
    }
    return route_map.get(state["route"], "draft_answer")


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(SupportState)
    builder.add_node("classify_request", classify_request)
    builder.add_node("ask_followup", ask_followup)
    builder.add_node("draft_answer", draft_answer)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("billing_helper", billing_helper)

    builder.add_edge(START, "classify_request")
    builder.add_conditional_edges("classify_request", choose_next_step)
    builder.add_edge("ask_followup", END)
    builder.add_edge("draft_answer", END)
    builder.add_edge("approval_gate", END)
    builder.add_edge("billing_helper", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    parser = argparse.ArgumentParser(description="Helpful Agent")
    parser.add_argument("query", help="Tell us what you need")

    args = parser.parse_args()

    result = app.invoke(
        {
            "user_input": args.query,
            "route": "",
            "response_text": "",
        }
    )
    print(result)
