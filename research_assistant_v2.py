import json
from enum import Enum
from typing import TypedDict, Literal
from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, field_validator

client = Groq()

TOPIC_KEYWORDS = {
    "pricing",
    "api",
    "refund",
    "shipping",
    "sdk",
    "sla",
    "plan",
    "policy",
    "authentication",
    "rate",
}

SUBJECT_KEYWORDS = {"it", "this", "that", "they", "them", "its"}


class CheckClarity(BaseModel):
    needs_clarification: bool
    clarification_question: str


class SearchDomainEnum(str, Enum):
    PRODUCT = "PRODUCT"
    TECHNICAL = "TECHNICAL"
    SUPPORT = "SUPPORT"

    @classmethod
    def list_domains(cls) -> list[str]:
        return [domain.value for domain in SearchDomainEnum]


class SearchPlan(BaseModel):
    sub_queries: list[str]

    @field_validator("sub_queries")
    @classmethod
    def valid_sub_queries(cls, value: list[str]):
        for sub_query in value:
            if ":" not in sub_query:
                raise ValueError(f"{sub_query} is not using the expected format.")
            domain = sub_query.split(":")[0].strip().upper()
            if domain not in SearchDomainEnum.list_domains():
                raise ValueError(f"Invalid domain: {domain}")
        return value


class ResearchState(TypedDict):
    # User input
    user_question: str

    # Clarity check - written by check_clarity
    needs_clarification: bool
    clarification_question: str

    # Guard
    max_steps: int
    step_count: int

    # Search planning - written by plan_searches
    search_plan: list[str]

    # Search execution - written by execute_searches
    search_results: list[str]
    sources_used: list[str]
    skipped_count: int

    # Quality
    quality_passed: bool
    quality_note: str

    # Synthesis - written by synthesize_findings
    synthesis: str
    confidence_level: str

    # Ouput - written by return_clarification or format_response
    formatted_response: str


def check_clarity(state: ResearchState) -> dict:
    question = state["user_question"].strip()
    words = question.split()
    words_lower = {w.lower().strip("?.,!") for w in words}

    if len(words) <= 3:
        return {
            "needs_clarification": True,
            "clarification_question": (
                f"Your question: '{question}' is too brief. "
                "Could you elaborate further?"
            ),
        }

    vague_openers = {"What's", "whats", "tell", "explain", "describe"}
    ambig_subjects = {
        "it",
        "this",
        "that",
        "these",
        "those",
        "everything",
        "stuff",
        "thing",
    }

    first_word = words[0].lower().rstrip("'s")
    second_word = words[1].lower()
    if first_word in vague_openers and second_word in ambig_subjects:
        return {
            "needs_clarification": True,
            "clarification_question": (
                "Your question references something without specifying the topic clearly. "
                "Could you name the specific feature, plan or policy you are asking about?"
            ),
        }

    has_pronoun = bool(words_lower & SUBJECT_KEYWORDS)
    has_topic = bool(words_lower & TOPIC_KEYWORDS)
    if has_pronoun and not has_topic:
        return {
            "needs_clarification": True,
            "clarification_question": (
                "Your question contains a pronoun but does not name a specific topic. "
                "Could you tell us which feature, plan or policy you are asking about?"
            ),
        }

    prompt = (
        "You are a research assistant intake filter.\n\n"
        "Decide whether the following question is specific enough to search a "
        "product knowledge base. A question is specific if it names a topic "
        "(pricing, API, refunds,shipping, SLA, SDK, authentication) and asks "
        "a clear question about it.\n\n"
        f"Question: {question}\n\n"
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "CheckClarity",
                "schema": CheckClarity.model_json_schema(),
            },
        },
    )
    check_clarity = CheckClarity.model_validate(
        json.loads(response.choices[0].message.content)
    )
    return check_clarity.model_dump()


def route_clarity(
    state: ResearchState,
) -> Literal["plan_searches", "return_clarification"]:
    return "return_clarification" if state["needs_clarification"] else "plan_searches"


def search_product_knowledge(query: str) -> str:

    articles = {
        "pricing": (
            "Plans: Start $9/mo (5 users), Pro $49/mo (25 users + API access), "
            "Enterprise $199/mo (unlimited users, dedicated support, SLA guarantee)."
        ),
        "plan": (
            "Start suits individuals and small teams. Pro unlocks the API and "
            "advanced analytics. Enterprise adds SSO, audit logs, and a 99.9% uptime SLA."
        ),
        "refund": (
            "Refund policy: full refunds within 30 days of purchase. "
            "Digital products have a 14-day refund window. Contact support2example.com."
        ),
        "shipping": (
            "Physical goods: standard delviery 5-7 business days. "
            "Express (2-3 days) costs $9.99. Free shipping on orders over $75."
        ),
        "api": (
            "API access is included on Pro and Enterprise plans. "
            "Rate limits: 1,000 requests/day (Pro), 10,000 requests/day (Enterprise). "
            "Authentication uses Bearer tokens."
        ),
    }
    q = query.lower()
    for keyword, content in articles.items():
        if keyword in q:
            return content
    return "No specific product information found. Please contact support for details."


def search_technical_docs(query: str) -> str:
    docs = {
        "sdk": (
            "SDKs available: Python (pip install example-sdk), "
            "JavaScript (npm install example-sdk), Go (go get example.com/sdk). "
            "All SDKs are open source. Minimum Python version: 3.9."
        ),
        "authentication": (
            "Authentication: Bearer token in the Authorization header. "
            "Generate tokens via the dashboard under Settings → API Keys. "
            "Tokens do not expire but can be revoked at any time."
        ),
        "oauth": (
            "OAuth 2.0 is supported on Enterprise plans. "
            "Supported flows: Authorization Code, Client Credentials. "
            "PKCE is required for public clients."
        ),
        "webhook": (
            "Webhooks: configure endpoint URLs in the dashboard. "
            "Events: payment.completed, subscription.cancelled, user.created. "
            "Payloads are signed with HMAC-SHA256. Retry policy: 3 attempts."
        ),
        "error": (
            "Common error codes: 400 Bad Request (malformed payload), "
            "401 Unauthorized (invalid or missing token), "
            "429 Too Many Requests (rate limit exceeded), "
            "500 Internal Server Error (contact support with the request ID)."
        ),
    }
    q = query.lower()
    for keyword, content in docs.items():
        if keyword in q:
            return content
    return "No specific technical documentation found. Check the developer portal for full docs."


def search_support_policies(query: str) -> str:
    policies = {
        "sla": (
            "SLA: Enterprise plan guarantees 99.9% monthly uptime. "
            "Compensation: 10% credit per 0.1% below the SLA threshold. "
            "Applies to API and dashboard availability."
        ),
        "support": (
            "Support tiers: Starter — community forum only. "
            "Pro — email support, 2-business-day response. "
            "Enterprise — dedicated Slack channel, 4-hour response, named support engineer."
        ),
        "response time": (
            "Response times: community forum (no SLA), "
            "email (2 business days for Pro), "
            "Slack/phone (4 hours for Enterprise)."
        ),
        "escalate": (
            "Escalation: if an issue is not resolved within the SLA window, "
            "reply to the ticket with 'ESCALATE' to trigger a senior engineer review. "
            "Enterprise customers can call the dedicated support line."
        ),
        "contact": (
            "Contact: support@example.com (general), "
            "billing@example.com (billing disputes), "
            "security@example.com (vulnerabilities). "
            "Phone support is available on Enterprise plans only."
        ),
    }
    q = query.lower()
    for keyword, content in policies.items():
        if keyword in q:
            return content
    return "No specific policy found. Contact support@example.com for guidance."


def plan_searches(state: ResearchState) -> dict:

    prompt = (
        "You are a search planner for a product knowledge assistant.\n\n"
        "Break the following research question into 2 or 3 focused sub-queries. "
        "For each sub-query, prefix it with the most relevant domain tag:\n"
        " PRODUCT: - pricing, plans, refunds, shipping, API access\n"
        " TECHNICAL: - SDK, authentication, Oauth, webhooks, error codes\n"
        " SUPPORT: - SLA, response times, escalation, contact channels\n\n"
        "Rules:\n"
        "- Use at most one sub-query per domain.\n"
        "- Only include domains genuinely relevant to the question.\n"
        "- Each sub-query must be a complete, searchable sentence.\n\n"
        f"Research question: {state['user_question']}\n\n"
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SearchPlan",
                "schema": SearchPlan.model_json_schema(),
            },
        },
    )
    search_plan = SearchPlan.model_validate(
        json.loads(response.choices[0].message.content)
    )
    return {
        "search_plan": search_plan.sub_queries,
        "step_count": len(search_plan.sub_queries),
    }


def return_clarification(state: ResearchState) -> dict:
    cq = state["clarification_question"].strip()
    return {
        "formatted_response": f"Before we search, I need a bit more detail:\n\n{cq}"
    }


def execute_searches(state: ResearchState) -> dict:
    results = []
    sources = []
    max_s = state["max_steps"]
    plan = state["search_plan"]
    skipped = max(0, len(plan) - max_s)

    tool_map = {
        "PRODUCT": (search_product_knowledge, "product_knowledge"),
        "TECHNICAL": (search_technical_docs, "technical_docs"),
        "SUPPORT": (search_support_policies, "support_policies"),
    }

    for i, entry in enumerate(state["search_plan"]):
        if i >= max_s:
            break
        if ":" not in entry:
            skipped += 1
            continue

        tag, sub_query = entry.split(":")
        tag = tag.strip().upper()
        sub_query = sub_query.strip()

        tool_fn, source_name = tool_map.get(
            tag, (search_product_knowledge, "product_knowledge")
        )
        result = tool_fn(sub_query)
        results.append(result)
        sources.append(source_name)

    return {"search_results": results, "sources_used": sources, "skipped": skipped}


def synthesize_findings(state: ResearchState) -> dict:
    return {"synthesis": "", "confidence_level": ""}


def format_response(state: ResearchState) -> dict:
    return {"formatted_response": ""}


def build_graph() -> CompiledStateGraph:

    builder = StateGraph(ResearchState)
    builder.add_node("check_clarity", check_clarity)
    builder.add_node("return_clarification", return_clarification)
    builder.add_node("plan_searches", plan_searches)
    builder.add_node("execute_searches", execute_searches)
    builder.add_node("synthesize_findings", synthesize_findings)
    builder.add_node("format_response", format_response)

    builder.add_edge(START, "check_clarity")
    builder.add_conditional_edges("check_clarity", route_clarity)
    builder.add_edge("plan_searches", "execute_searches")
    builder.add_edge("execute_searches", "synthesize_findings")
    builder.add_edge("synthesize_findings", "format_response")
    builder.add_edge("format_response", END)
    builder.add_edge("return_clarification", END)

    return builder.compile()


def pretty_print(question: str, result: dict):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f" Question: {question[:80]}")
    print(bar)
    print(f" needs clarification: {result['needs_clarification']}")
    if result["needs_clarification"]:
        print(f" clarification asked: {result['clarification_question'][:200]}")
        return
    print(f" search plan : {result['search_plan']}")
    print(f" step_count: {result['step_count']}")
    print(f" sources_used: {result['sources_used']}")
    for i, (src, res) in enumerate(
        zip(result["sources_used"], result["search_results"])
    ):
        print(f" result {i}: [{src}]: {res[:100]}")
    print(f" synthesis: {result['synthesis']}")
    print(f" formatted response: {result['formatted_response'][:200]}")


if __name__ == "__main__":
    app = build_graph()

    test_questions = [
        "api",
        "Tell me everything",
        "What's the deal with your service?",
        "What is the pricing for the Pro plan and what are the API rate limits?",
        "What SDK is available and what level of support do Pro users get?",
    ]

    for question in test_questions:
        result = app.invoke(
            {
                "user_question": question,
                "needs_clarification": False,
                "clarification_question": "",
                "search_plan": [],
                "max_steps": 3,
                "step_count": 0,
                "search_results": [],
                "sources_used": [],
                "synthesis": "",
                "confidence_level": "",
                "formatted_response": "",
            }
        )
        pretty_print(question=question, result=result)
