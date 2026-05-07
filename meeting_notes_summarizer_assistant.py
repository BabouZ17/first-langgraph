from typing import TypedDict, Literal
from google.genai import Client, types
import argparse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Client()


class MeetingState(TypedDict):
    # User input (immutable)
    meeting_transcript: str
    meeting_title: str

    # Control
    input_valid: bool
    quality_passed: bool

    # Context
    word_count: int

    # Output
    raw_summary: str
    final_summary: str
    error_note: str


def validate_input(state: MeetingState) -> dict:
    meeting_transcript = state["meeting_transcript"].strip()
    word_count = len(meeting_transcript.split())
    if not state["meeting_title"].strip():
        return {
            "input_valid": False,
            "word_count": word_count,
            "error_note": "Invalid meeting title.",
        }
    if word_count < 30:
        return {
            "intput_valid": False,
            "word_count": word_count,
            "error_note": "Not enough words in meeting_transcript.",
        }
    return {"input_valid": True, "word_count": word_count, "error_note": ""}


def draft_summary(state: MeetingState) -> dict:
    prompt = (
        f"You are to summarize notes from a meeting title '{state['meeting_title']}'.\n"
        "Write a summary in 4-6 sentences covering:\n"
        "- The main topics discusses\n"
        "- Any decisions made\n"
        "- Any action items mentioned\n\n"
        f"Meeting notes:\n{state['meeting_transcript']}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )
    return {"raw_summary": response.text}


def check_quality(state: MeetingState) -> dict:
    summary = state["raw_summary"].strip()
    word_count = len(summary.split())
    has_content = word_count >= 20
    is_not_empty = len(summary) >= 60
    return {"quality_passed": has_content and is_not_empty}


def finalize_summary(state: MeetingState) -> dict:
    header = (
        f"Meeting Summary\n"
        f"{'=' * 40}\n"
        f"Title : {state['meeting_title']}\n"
        f"Word count: {state['word_count']} word in meeting_transcript.\n"
        f"{'=' * 40}\n\n"
    )
    body = state["raw_summary"].strip()
    final_summary = header + body
    return {"final_summary": final_summary}


def route_after_validation(state: MeetingState) -> Literal["draft_summary", "__end__"]:
    return "draft_summary" if state["input_valid"] else "__end__"


def route_after_quality(state: MeetingState) -> Literal["finalize_summary", "__end__"]:
    return "finalize_summary" if state["quality_passed"] else "__end__"


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(MeetingState)

    builder.add_node("validate_input", validate_input)
    builder.add_node("draft_summary", draft_summary)
    builder.add_node("check_quality", check_quality)
    builder.add_node("finalize_summary", finalize_summary)

    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges("validate_input", route_after_validation)
    builder.add_edge("draft_summary", "check_quality")
    builder.add_conditional_edges("check_quality", route_after_quality)
    builder.add_edge("finalize_summary", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    parser = argparse.ArgumentParser(description="Meeting Notes Summarizer Assistant")
    parser.add_argument("--meeting-title", help="Title of the meeting")
    parser.add_argument("--meeting-transcript", help="Meeting transcript")

    args = parser.parse_args()

    result = app.invoke(
        {
            "meeting_transcript": args.meeting_transcript,
            "meeting_title": args.meeting_title,
            "input_valid": False,
            "quality_passed": False,
            "word_count": 0,
            "raw_summary": "",
            "final_summary": "",
            "error_note": "",
        }
    )
    print(result)
