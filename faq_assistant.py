from typing import TypedDict, Literal
from google.genai import Client, types
import argparse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Client()


class FAQState(TypedDict):
    user_question: str
    route: str
    context: str
    draft_response: str
    quality_passed: bool


def classify_message(state: FAQState) -> dict:
    text = state["user_question"].lower()
    if any(
        word in text
        for word in ["account", "bill", "charge", "invoice", "subscription"]
    ):
        return {"route": "account"}
    elif any(word in text for word in ["error", "bug", "crash", "not working"]):
        return {"route": "technical"}
    return {"route": "general"}


def technical_triage(state: FAQState) -> dict:
    return {"draft_response": ("Please share your error message and platform details")}


def retrieve_context(state: FAQState) -> dict:
    question = state["user_question"]
    context = (
        f"Account context for query '{question}': "
        "Customer is on the Pro plan. Last invoice was paid on time. "
        "No outstanding charges."
    )
    return {"context": context}


def draft_response(state: FAQState) -> dict:
    context_block = ""
    if state["context"]:
        context_block = f"\n\nRelevant context:\n{state['context']}"

    prompt = (
        "You are a helpful FAQ Assistant. "
        "Answer the following question clearly in 3-5 lines."
        f"{context_block}\n\n"
        f"Question: {state['user_question']}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )
    return {"draft_response": response.text}


def quality_gate(state: FAQState) -> dict:
    draft = state["draft_response"]
    passed = len(draft.strip()) >= 40
    return {"quality_passed": passed}


def route_after_classify(
    state: FAQState,
) -> Literal["retrieve_context", "draft_response", "technical_triage"]:
    mapping = {
        "account": "retrieve_context",
        "technical": "technical_triage",
        "general": "draft_response",
    }
    return mapping.get(state["route"], "draft_response")


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(FAQState)
    builder.add_node("classify_message", classify_message)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("draft_response", draft_response)
    builder.add_node("quality_gate", quality_gate)
    builder.add_node("technical_triage", technical_triage)

    builder.add_edge(START, "classify_message")
    builder.add_conditional_edges("classify_message", route_after_classify)
    builder.add_edge("retrieve_context", "draft_response")
    builder.add_edge("draft_response", "quality_gate")
    builder.add_edge("quality_gate", END)
    builder.add_edge("technical_triage", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    parser = argparse.ArgumentParser(description="Helpful FAQ Agent")
    parser.add_argument("query", help="Tell us what you need")

    args = parser.parse_args()

    result = app.invoke(
        {
            "user_question": args.query,
            "route": "",
            "context": "",
            "draft_response": "",
            "quality_passed": False,
        }
    )
    print(result)
