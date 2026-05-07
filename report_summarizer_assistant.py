from typing import TypedDict
from google.genai import Client, types
import argparse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Client()


class ReportState(TypedDict):
    # User input (immutable)
    report_text: str

    # Classification
    doc_type: str
    retry_count: int

    # Context
    word_count: int
    policy_context: str

    # Output
    summary: str
    confidence: float
    error_message: str
    quality_passed: bool


def classify_report(state: ReportState) -> dict:
    text = state["report_text"]
    word_count = len(text.split())
    if any(w in text.lower() for w in ["financial", "revenue", "profit", "loss"]):
        doc_type = "financial"
    elif any(
        w in text.lower() for w in ["technical", "system", "infrastructur", "api"]
    ):
        doc_type = "technical"
    else:
        doc_type = "general"
    return {"doc_type": doc_type, "word_count": word_count}


def retrieval_policies(state: ReportState) -> dict:
    policies = {
        "financial": (
            "Policy: All financial summaries must include key metrics and risk flags."
        ),
        "technical": (
            "Policy: Technical summaries must reference system components and impact."
        ),
        "general": ("Policy: Keep summaries concise, fatcual and action-oriented."),
    }
    return {
        "policy_context": policies.get(state["policy_context"], policies["general"])
    }


def draft_summary(state: ReportState) -> dict:
    prompt = (
        "Summarize the following report in 3-5 sentences.\n\n"
        f"Guidance: {state['policy_context']}\n\n"
        f"Report:\n{state['report_text']}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )
    return {"summary": response.text}


def quality_gate(state: ReportState) -> dict:
    summary, report = state["summary"], state["report_text"]
    passed = len(summary.strip()) >= 80 and len(summary.split()) >= 15
    confidence = round(len(summary.split()) / max(len(report.split()), 1), 2)
    output = {
        "quality_passed": passed,
        "retry_count": state["retry_count"] + 1,
        "confidence": confidence,
    }
    if not passed:
        output.update(
            {"error_message": "Quality of produced summary is not good enough."}
        )
    return output


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(ReportState)

    builder.add_node("classify_report", classify_report)
    builder.add_node("retrieve_policies", retrieval_policies)
    builder.add_node("draft_summary", draft_summary)
    builder.add_node("quality_gate", quality_gate)

    builder.add_edge(START, "classify_report")
    builder.add_edge("classify_report", "retrieve_policies")
    builder.add_edge("retrieve_policies", "draft_summary")
    builder.add_edge("draft_summary", "quality_gate")
    builder.add_edge("quality_gate", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    parser = argparse.ArgumentParser(description="Report Summarizer Assistant")
    parser.add_argument("query", help="Tell us what you need")

    args = parser.parse_args()

    result = app.invoke(
        {
            "report_text": args.query,
            "doc_type": "",
            "retry_count": 0,
            "word_count": 0,
            "policy_context": "",
            "summary": "",
            "confidence": 0.0,
            "error_message": "",
            "quality_passed": False,
        }
    )
    print(result)
