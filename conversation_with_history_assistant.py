from typing import TypedDict
from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = Groq()


class ConversationState(TypedDict):
    user_message: str
    messages: list[str]
    response: str


def add_user_message(state: ConversationState) -> dict:
    new_entry = {"role": "user", "content": state["user_message"]}
    return {"messages": state["messages"] + [new_entry]}


def add_assistant_message(state: ConversationState) -> dict:
    new_entry = {"role": "assistant", "content": state["response"]}
    return {"messages": state["messages"] + [new_entry]}


def generate_response(state: ConversationState) -> dict:

    system_msg = {
        "role": "system",
        "content": "You are a helpful product support assistant. Respond in 2-3 sentences.",
    }
    history = state["messages"]
    messages = (
        [system_msg] + history + [{"role": "user", "content": state["user_message"]}]
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages
    )
    return {"response": response.choices[0].message.content}


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(ConversationState)
    builder.add_node("add_user_message", add_user_message)
    builder.add_node("add_assistant_message", add_assistant_message)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "add_user_message")
    builder.add_edge("add_user_message", "generate_response")
    builder.add_edge("generate_response", "add_assistant_message")
    builder.add_edge("add_assistant_message", END)

    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph()

    config = {"configurable": {"thread_id": "bob"}}

    result = app.invoke(
        {"user_message": "What is your refund policy?", "messages": [], "response": ""},
        config=config,
    )
    print(result)

    result = app.invoke(
        {"user_message": "How long does the refund take?"}, config=config
    )
    print(result)

for snapshot in app.get_state_history(config):
    node = snapshot.metadata.get("source", "unknown")
    print(f"\nAfter node: {node}")
    print(f" user_message: {snapshot.values.get('user_message', 'not yet written')}")
    print(f" messages: {snapshot.values.get('messages', 'not yet written')}")
    print(f" response: {snapshot.values.get('response', 'not yet_written')}")
