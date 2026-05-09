from typing import TypedDict, Literal
from google.genai import Client, types
import argparse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Client()


class SupportState(TypedDict):
    # User input (immutable)
    user_message: str

    # Control
    intent: str
    confidence: str

    # Output
    response: str


def classifiy_intent(state: SupportState) -> dict:
    text = state["user_message"].lower()

    urgent_keywords = ["urgent", "asap", "emergency", "immediately"]
    if any(kw in text for kw in urgent_keywords):
        return {"intent": "urgent", "confidence": "high"}

    intent_keywords = {
        "faq": ["how", "what", "when", "does", "can", "feature", "work", "explain"],
        "billing": [
            "invoice",
            "charge",
            "payment",
            "refund",
            "subscription",
            "price",
            "cost",
        ],
        "escalation": [
            "complaint",
            "unacceptable",
            "manager",
            "legal",
            "terrible",
            "lawsuit",
        ],
    }

    scores = {
        intent: sum(1 for kw in keywords if kw in text)
        for intent, keywords in intent_keywords.items()
    }

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score == 0:
        return {"intent": "fallback", "confidence": "low"}
    confidence = "high" if best_score >= 2 else "low"
    return {"intent": best_intent, "confidence": confidence}


def handle_urgent(state: SupportState) -> dict:
    return {
        "response": (
            "Your request has been flagged as urgent. "
            "A team member will review it within one hour and reach out to you directly."
        )
    }


def handle_faq(state: SupportState) -> dict:
    prompt = (
        "You are a helful customer support assistant.\n"
        "Answer the following question clearly in 3-4 sentences.\n"
        f"Customer question: {state['user_message']}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )
    return {"response": response.text}


def handle_billing(state: SupportState) -> dict:
    return {
        "response": (
            "For billing and payment questions, our team is ready to help. "
            "Please share your account email and the invoice or transaction reference, "
            "and will look into this within one business day."
        )
    }


def handle_escalation(state: SupportState) -> dict:
    return {
        "response": (
            "We take your concern seriously and want to make sure it receives proper attention. "
            "A senior team member will review your message and follow up within 24 hours."
        )
    }


def handle_fallback(state: SupportState) -> dict:
    return {
        "response": (
            "I want to make sure I direct your question to the right place. "
            "Could you describe what you need with in a bit more detail?"
        )
    }


def route_by_intent(
    state: SupportState,
) -> Literal[
    "handle_faq",
    "handle_billing",
    "handle_escalation",
    "handle_fallback",
    "handle_urgent",
]:
    if state["confidence"] == "low":
        return "handle_fallback"

    routes = {
        "faq": "handle_faq",
        "billing": "handle_billing",
        "escalation": "handle_escalation",
        "urgent": "handle_urgent",
        "fallback": "handle_fallback",
    }
    return routes.get(state["intent"], "handle_fallback")


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(SupportState)

    builder.add_node("classify_intent", classifiy_intent)
    builder.add_node("handle_faq", handle_faq)
    builder.add_node("handle_billing", handle_billing)
    builder.add_node("handle_escalation", handle_escalation)
    builder.add_node("handle_urgent", handle_urgent)
    builder.add_node("handle_fallback", handle_fallback)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent)
    builder.add_edge("handle_urgent", END)
    builder.add_edge("handle_faq", END)
    builder.add_edge("handle_billing", END)
    builder.add_edge("handle_escalation", END)
    builder.add_edge("handle_fallback", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    parser = argparse.ArgumentParser(description="Customer Support Assistant")
    parser.add_argument("--query", help="Your query")

    args = parser.parse_args()

    result = app.invoke(
        {"user_message": args.query, "intent": "", "confidence": "", "response": ""}
    )
    print(result)
