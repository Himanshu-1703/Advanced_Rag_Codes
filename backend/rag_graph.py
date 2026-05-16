import os
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import ClaimVerificationResult, RelevancyDecision, RouterDecision
from backend.vector_store import search as vs_search

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "When in doubt between retrieve and direct_answer, prefer retrieve.\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    decision: RouterDecision = router_chain.invoke({"query": query})
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message history.
    # retrieval llm --> tool call --> tool result
    # llm --> no tools are bounded --> tool call
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm
    messages = [{"role": "system", "content": RETRIEVE_SYSTEM}] + state["messages"]
    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = llm.with_structured_output(ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # arXiv-targeted search via web to get paper titles and links
    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Build context block
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't know the answer."
            else:
                context = "\n\n---\n\n".join(doc.page_content for doc in docs)
                prompt = f"Answer the question using this context:\n\n{context}\n\nQuestion: {query}"
                answer = llm.invoke([{"role": "user", "content": prompt}]).content

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        prompt = f"Answer from your knowledge.\n\nQuestion: {query}"
        answer = llm.invoke([{"role": "user", "content": prompt}]).content

    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)

