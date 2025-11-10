"""FastAPI entrypoint for the QA service."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, RedirectResponse

from .config import load_settings
from .embeddings import EmbeddingsClient
from .llm import LLMClient
from .message_client import MessageRecord, MessagesClient
from .schemas import AnswerResponse, MessageSchema
from .service import QAService


settings = load_settings()
messages_client = MessagesClient(settings.messages_api)
embeddings_client = EmbeddingsClient(settings.openai)
llm_client = LLMClient(settings.openai)
qa_service = QAService(
    messages_client,
    embeddings_client,
    llm_client,
    retrieval_top_k=settings.retrieval.top_k,
    message_cache_limit=settings.messages_api.limit,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await qa_service.warm_cache()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
MESSAGE_LIST_LIMIT = 50
REASONING_CHOICES = {"minimal", "low", "medium", "high"}


def _serialize_message(record: MessageRecord) -> MessageSchema:
    return MessageSchema(
        user_name=record.user_name,
        timestamp=record.timestamp,
        message=record.message,
    )


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/home", status_code=307)


@app.get("/home", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    html = f"""
    <html>
        <head><title>{settings.app_name}</title></head>
        <body>
            <style>
                body {{
                    margin: 0;
                    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                    background: #f1f5f9;
                    color: #0f172a;
                }}
                header {{
                    background: linear-gradient(120deg, #2563eb, #7c3aed);
                    color: #fff;
                    padding: 80px 32px;
                    text-align: center;
                }}
                header h1 {{
                    margin: 0;
                    font-size: 2.8rem;
                }}
                header p {{
                    margin-top: 16px;
                    font-size: 1.2rem;
                    opacity: 0.9;
                }}
                main {{
                    padding: 40px;
                    max-width: 1100px;
                    margin: 0 auto;
                }}
                .card-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                    gap: 20px;
                }}
                .card {{
                    background: #fff;
                    border-radius: 16px;
                    padding: 24px;
                    box-shadow: 0 12px 25px rgba(15, 23, 42, 0.08);
                }}
                .card h3 {{ margin-top: 0; }}
                .card a {{ color: #2563eb; font-weight: 600; text-decoration: none; }}
                .section {{
                    margin-top: 48px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: #fff;
                    border-radius: 14px;
                    overflow: hidden;
                    box-shadow: 0 10px 20px rgba(15,23,42,0.05);
                }}
                th, td {{
                    padding: 14px 18px;
                    border-bottom: 1px solid #e2e8f0;
                    text-align: left;
                }}
                th {{
                    background: #f8fafc;
                    font-size: 0.95rem;
                    text-transform: uppercase;
                    letter-spacing: .04em;
                }}
                tr:last-child td {{ border-bottom: none; }}
                code {{
                    background: #0f172a;
                    color: #e2e8f0;
                    padding: 6px 10px;
                    border-radius: 8px;
                    display: inline-block;
                }}
            </style>
            <header>
                <h1>{settings.app_name}</h1>
                <p>Concierge-grade Q&A on top of the member message stream. Ask, explore, or plug it into your own workflows.</p>
            </header>
            <main>
                <section class="card-grid">
                    <div class="card">
                        <h3>Live Demo</h3>
                        <p>The split-screen UI shows the chat interface alongside the retrieved snippets powering each answer.</p>
                        <a href="/demo">Open demo →</a>
                    </div>
                </section>

                <section class="section">
                    <h2>Endpoint Directory</h2>
                    <table>
                        <tr><th>Method</th><th>Path</th><th>Description</th></tr>
                        <tr><td>GET</td><td>/ask</td><td>Primary question-answering endpoint (query parameter <code>question=</code>).</td></tr>
                        <tr><td>GET</td><td>/messages</td><td>Paginated member messages with optional <code>?limit=</code>.</td></tr>
                        <tr><td>GET</td><td>/demo</td><td>Interactive chat + context viewer.</td></tr>
                        <tr><td>GET</td><td>/home</td><td>This overview page.</td></tr>
                    </table>
                </section>

                <section class="section">
                    <h2>Quick Start</h2>
                    <p>Ask a question directly from your terminal:</p>
<code>curl "http://localhost:8000/ask?question=Who%20needs%20payment%20help%3F"</code>
                </section>
            </main>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/messages")
async def list_messages(
    limit: int = Query(
        MESSAGE_LIST_LIMIT,
        ge=1,
        le=500,
        description="Maximum number of messages to return",
    )
) -> Response:
    try:
        requested = min(limit, settings.messages_api.limit)
        records = await messages_client.fetch_messages(limit=requested)
        payload = [
            _serialize_message(record).model_dump(mode="json") for record in records
        ]
        formatted = json.dumps(payload, indent=2)
        return Response(content=formatted, media_type="application/json")
    except Exception as exc: 
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/ask", response_model=AnswerResponse)
async def ask(
    question: str = Query(..., min_length=1, description="Natural language question"),
    reasoning: str
    | None = Query(
        default=None,
        description="Optional reasoning effort override (minimal|low|medium|high).",
    ),
) -> AnswerResponse:
    normalized = _normalize_question(question)
    effort = _normalize_reasoning(reasoning)
    return await _answer_question(normalized, effort)


async def _answer_question(
    question: str, reasoning_effort: str | None = None
) -> AnswerResponse:
    try:
        answer, count = await qa_service.answer_question(
            question, reasoning_effort=reasoning_effort
        )
        return AnswerResponse(answer=answer, sources_used=count)
    except Exception as exc:  # pragma: no cover - FastAPI will handle logging
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _normalize_question(question: str) -> str:
    value = question.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    if not value:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    return value


def _normalize_reasoning(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered not in REASONING_CHOICES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid reasoning effort '{value}'. "
                f"Choose from {', '.join(sorted(REASONING_CHOICES))}."
            ),
        )
    return lowered


@app.get("/demo", response_class=HTMLResponse)
async def demo() -> HTMLResponse:
    cached_messages = await qa_service.get_cached_messages()
    serialized_cache = [
        _serialize_message(record).model_dump(mode="json") for record in cached_messages
    ]
    safe_cache_json = json.dumps(serialized_cache).replace("</", "<\\/")
    default_reasoning = settings.openai.reasoning_effort.capitalize()
    html = f"""
    <html>
        <head>
            <title>{settings.app_name} Demo</title>
            <style>
                :root {{
                    color-scheme: light;
                    --bg: #f2f5fb;
                    --card: #ffffff;
                    --border: #d8e1f0;
                    --primary: #2563eb;
                    --primary-dark: #1d4ed8;
                    --question-bg: #e0ebff;
                    --answer-bg: #e8f9f1;
                }}
                * {{
                    box-sizing: border-box;
                }}
                body {{
                    margin: 0;
                    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                    background: var(--bg);
                    color: #0f172a;
                    min-height: 100vh;
                }}
                h2 {{
                    margin-top: 0;
                }}
                .container {{
                    display: flex;
                    gap: 20px;
                    padding: 32px;
                    height: 100vh;
                }}
                .panel {{
                    flex: 1;
                    background: var(--card);
                    border-radius: 18px;
                    padding: 24px;
                    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
                    display: flex;
                    flex-direction: column;
                }}
                .chat-panel {{
                    border: 1px solid var(--border);
                }}
                .chat-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                    gap: 16px;
                    margin-bottom: 18px;
                }}
                .chat-title {{
                    margin: 0;
                }}
                .messages-panel {{
                    border: 1px solid var(--border);
                }}
                .chat-log {{
                    list-style: none;
                    padding: 0;
                    margin: 0 0 16px 0;
                    overflow-y: auto;
                    flex: 1;
                }}
                .chat-log li {{
                    margin-bottom: 12px;
                    padding: 14px 16px;
                    border-radius: 12px;
                    line-height: 1.4;
                    position: relative;
                }}
                .chat-log li p {{
                    margin: 6px 0 0;
                    white-space: pre-line;
                }}
                .chat-log li .message-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.78rem;
                    letter-spacing: 0.04em;
                    text-transform: uppercase;
                    margin-bottom: 6px;
                    opacity: 0.75;
                }}
                .chat-log li.question .message-header span {{
                    color: #1e3a8a;
                }}
                .chat-log li.answer .message-header span {{
                    color: #047857;
                }}
                .message-timing {{
                    font-size: 0.75rem;
                    color: #475569;
                    margin-left: auto;
                    text-transform: none;
                }}
                .message-action {{
                    border: none;
                    background: transparent;
                    color: #2563eb;
                    font-weight: 600;
                    font-size: 0.78rem;
                    cursor: pointer;
                }}
                .message-action:hover {{
                    text-decoration: underline;
                }}
                .message-action:disabled {{
                    opacity: 0.5;
                    cursor: not-allowed;
                }}
                .chat-log li.editing {{
                    outline: 2px dashed rgba(37, 99, 235, 0.5);
                    background: rgba(219, 234, 254, 0.7);
                }}
                .chat-log li.pending p {{
                    position: relative;
                    color: #0f172a;
                    overflow: hidden;
                }}
                .chat-log li.pending p::after {{
                    content: '';
                    position: absolute;
                    inset: 0;
                    background: linear-gradient(120deg,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.65) 45%,
                        rgba(255, 255, 255, 0) 70%);
                    transform: translateX(-100%);
                    animation: textShimmer 2.2s linear infinite;
                    mix-blend-mode: screen;
                    filter: blur(0.5px);
                }}
                @keyframes textShimmer {{
                    0% {{
                        transform: translateX(-130%);
                    }}
                    100% {{
                        transform: translateX(130%);
                    }}
                }}
                .chat-log .question {{
                    align-self: flex-end;
                    background: var(--question-bg);
                    border: 1px solid rgba(37, 99, 235, 0.15);
                }}
                .chat-log .answer {{
                    background: var(--answer-bg);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                }}
                .input-row {{
                    display: flex;
                    gap: 12px;
                    align-items: center;
                }}
                .reasoning-control {{
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    gap: 6px;
                    min-width: 210px;
                }}
                .reasoning-control label {{
                    font-size: 0.72rem;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    color: #475569;
                    font-weight: 600;
                }}
                .reasoning-dropdown {{
                    position: relative;
                    width: 100%;
                }}
                .reasoning-toggle {{
                    width: 100%;
                    border: 1px solid rgba(148, 163, 184, 0.5);
                    border-radius: 18px;
                    background: linear-gradient(145deg, #ffffff, #f2f6ff);
                    color: #0f172a;
                    font-size: 0.95rem;
                    font-weight: 600;
                    padding: 10px 44px 10px 16px;
                    text-align: left;
                    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
                    cursor: pointer;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 12px;
                    transition: border 0.2s, box-shadow 0.2s, transform 0.2s;
                }}
                .reasoning-toggle:focus-visible {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18);
                    transform: translateY(-1px);
                }}
                .reasoning-toggle svg {{
                    width: 16px;
                    height: 16px;
                    fill: none;
                    stroke: #475569;
                    stroke-width: 2;
                }}
                .reasoning-menu {{
                    position: absolute;
                    top: calc(100% + 8px);
                    right: 0;
                    width: 100%;
                    margin: 0;
                    padding: 8px;
                    list-style: none;
                    background: #fff;
                    border-radius: 18px;
                    border: 1px solid rgba(148, 163, 184, 0.4);
                    box-shadow: 0 18px 30px rgba(15, 23, 42, 0.18);
                    opacity: 0;
                    pointer-events: none;
                    transform: translateY(-6px);
                    transition: opacity 0.2s ease, transform 0.2s ease;
                    z-index: 20;
                }}
                .reasoning-dropdown[data-open="true"] .reasoning-menu {{
                    opacity: 1;
                    pointer-events: auto;
                    transform: translateY(0);
                }}
                .reasoning-option {{
                    width: 100%;
                    border: none;
                    background: transparent;
                    border-radius: 14px;
                    padding: 10px 12px;
                    font-size: 0.95rem;
                    font-weight: 500;
                    text-align: left;
                    color: #0f172a;
                    cursor: pointer;
                    transition: background 0.15s ease, color 0.15s ease;
                }}
                .reasoning-option:hover {{
                    background: rgba(37, 99, 235, 0.08);
                }}
                .reasoning-option.active {{
                    background: rgba(37, 99, 235, 0.15);
                    color: #1d4ed8;
                }}
                .input-row input {{
                    flex: 1;
                    height: 58px;
                    padding: 0 18px;
                    border-radius: 16px;
                    border: 1px solid var(--border);
                    font-size: 1rem;
                    line-height: 1;
                    transition: border 0.2s, box-shadow 0.2s;
                }}
                .input-row input:focus {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
                }}
                .send-stop-button {{
                    width: 58px;
                    height: 58px;
                    border-radius: 18px;
                    border: none;
                    padding: 0;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    background: linear-gradient(145deg, #1f2937, #0b1221 65%);
                    box-shadow: inset 0 2px 6px rgba(255, 255, 255, 0.25), inset 0 -6px 16px rgba(11, 17, 32, 0.9), 0 10px 25px rgba(15, 23, 42, 0.4);
                    cursor: pointer;
                    transition: transform 0.15s ease, filter 0.2s ease;
                }}
                .send-stop-button::after {{
                    content: '';
                    position: absolute;
                    inset: 5px;
                    border-radius: 14px;
                    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.35), rgba(15,23,42,0.85));
                    pointer-events: none;
                    z-index: 0;
                }}
                .send-stop-button:hover {{
                    filter: brightness(1.05);
                }}
                .send-stop-button:active {{
                    transform: translateY(1px);
                }}
                .send-stop-button[data-mode="stop"] {{
                    background: linear-gradient(145deg, #7f1d1d, #450a0a 70%);
                    box-shadow: inset 0 2px 8px rgba(255, 255, 255, 0.25), inset 0 -6px 18px rgba(69, 10, 10, 0.85), 0 10px 25px rgba(185, 28, 28, 0.35);
                }}
                .send-stop-button[data-mode="stop"]::after {{
                    background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.3), rgba(120, 15, 15, 0.9));
                }}
                .send-stop-button .icon {{
                    width: 22px;
                    height: 22px;
                    fill: #f8fafc;
                    opacity: 0;
                    transform: scale(0.6);
                    transition: opacity 0.12s ease, transform 0.18s ease;
                    position: relative;
                    z-index: 1;
                    display: block;
                    margin: 0;
                }}
                .send-stop-button[data-mode="send"] .icon-send,
                .send-stop-button[data-mode="stop"] .icon-stop {{
                    opacity: 1;
                    transform: scale(1);
                }}
                .status {{
                    min-height: 22px;
                    color: #475569;
                    margin-top: 10px;
                    font-size: 0.95rem;
                }}
                .status-row {{
                    display: flex;
                    gap: 12px;
                    align-items: center;
                    justify-content: space-between;
                    flex-wrap: wrap;
                }}
                .link-button {{
                    border: none;
                    background: transparent;
                    color: var(--primary);
                    font-weight: 600;
                    cursor: pointer;
                    padding: 0;
                }}
                .link-button:hover {{
                    text-decoration: underline;
                }}
                .link-button[hidden] {{
                    display: none;
                }}
                .sr-only {{
                    position: absolute;
                    width: 1px;
                    height: 1px;
                    padding: 0;
                    margin: -1px;
                    overflow: hidden;
                    clip: rect(0, 0, 0, 0);
                    white-space: nowrap;
                    border: 0;
                }}
                .status-secondary {{
                    margin-top: 8px;
                    font-size: 0.9rem;
                }}
                #messages-list {{
                    overflow-y: auto;
                    flex: 1;
                    border-radius: 12px;
                    border: 1px solid var(--border);
                    background: linear-gradient(180deg, #fff, #f8fbff);
                    padding: 0 12px;
                }}
                .message-card {{
                    padding: 16px 12px;
                    border-bottom: 1px solid rgba(148, 163, 184, 0.3);
                }}
                .message-card:last-child {{
                    border-bottom: none;
                }}
                .message-card strong {{
                    font-size: 1rem;
                    color: #0f172a;
                }}
                .message-card time {{
                    display: block;
                    font-size: 0.85rem;
                    color: #64748b;
                    margin-bottom: 6px;
                }}
                .message-card p {{
                    margin: 6px 0 0;
                }}
                .secondary-btn {{
                    margin-top: 16px;
                    align-self: flex-start;
                    padding: 10px 18px;
                    border-radius: 10px;
                    border: 1px solid var(--border);
                    background: #f8fafc;
                    color: #0f172a;
                    font-weight: 600;
                    cursor: pointer;
                    transition: background 0.2s, color 0.2s;
                }}
                .secondary-btn.loading {{
                    background: var(--primary);
                    color: #fff;
                    position: relative;
                    padding-right: 36px;
                }}
                .secondary-btn.loading::after {{
                    content: "";
                    position: absolute;
                    top: 50%;
                    right: 12px;
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    border: 2px solid rgba(255, 255, 255, 0.5);
                    border-top-color: #fff;
                    transform: translateY(-50%);
                    animation: spin 0.8s linear infinite;
                }}
                .secondary-btn:hover {{
                    background: #eef2ff;
                    color: var(--primary-dark);
                }}
                .secondary-btn:disabled {{
                    opacity: 0.6;
                    cursor: not-allowed;
                }}
                .secondary-btn.pulse {{
                    animation: btnPulse 0.3s ease;
                }}
                @keyframes btnPulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.03); }}
                    100% {{ transform: scale(1); }}
                }}
                @keyframes spin {{
                    0% {{ transform: translateY(-50%) rotate(0deg); }}
                    100% {{ transform: translateY(-50%) rotate(360deg); }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="panel chat-panel">
                    <div class="chat-header">
                        <h2 class="chat-title">Ask Concierge</h2>
                        <div class="reasoning-control">
                            <label for="reasoning-toggle">Reasoning</label>
                            <div class="reasoning-dropdown" data-open="false">
                                <input type="hidden" id="reasoning-effort" value="">
                                <button type="button" id="reasoning-toggle" class="reasoning-toggle" aria-haspopup="listbox" aria-controls="reasoning-menu" aria-expanded="false">
                                    <span id="reasoning-label">Default ({default_reasoning})</span>
                                    <svg viewBox="0 0 24 24" aria-hidden="true">
                                        <path d="M6 9l6 6 6-6" stroke-linecap="round" stroke-linejoin="round" />
                                    </svg>
                                </button>
                                <ul id="reasoning-menu" class="reasoning-menu" role="listbox">
                                    <li><button type="button" class="reasoning-option active" data-value="" data-label="Default ({default_reasoning})">Default ({default_reasoning})</button></li>
                                    <li><button type="button" class="reasoning-option" data-value="minimal" data-label="Minimal">Minimal</button></li>
                                    <li><button type="button" class="reasoning-option" data-value="low" data-label="Low">Low</button></li>
                                    <li><button type="button" class="reasoning-option" data-value="medium" data-label="Medium">Medium</button></li>
                                    <li><button type="button" class="reasoning-option" data-value="high" data-label="High">High</button></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <ul id="chat-log" class="chat-log"></ul>
                    <form id="ask-form" class="input-row">
                        <input id="question-input" type="text" placeholder="Ask anything about the members..."
                            autocomplete="off" />
                        <button id="send-stop-button" class="send-stop-button" type="submit" data-mode="send"
                            data-intent="send" aria-label="Send question">
                            <span class="sr-only">Send</span>
                            <svg class="icon icon-send" viewBox="0 0 24 24" aria-hidden="true" focusable="false"
                                preserveAspectRatio="xMidYMid meet">
                                <path d="M12 4l-6 6h4v10h4V10h4z" />
                            </svg>
                            <svg class="icon icon-stop" viewBox="0 0 24 24" aria-hidden="true" focusable="false"
                                preserveAspectRatio="xMidYMid meet">
                                <path d="M7 7h10v10H7z" />
                            </svg>
                        </button>
                    </form>
                    <div class="status-row">
                        <div id="status" class="status">Ready. Stop a response to branch or edit earlier prompts.</div>
                        <button type="button" id="cancel-edit" class="link-button" hidden>Cancel edit</button>
                    </div>
                </div>
                <div class="panel messages-panel">
                    <h2>Latest Member Messages</h2>
                    <div id="messages-list">Loading messages...</div>
                    <button id="show-more" class="secondary-btn">Show more</button>
                    <div id="messages-status" class="status status-secondary"></div>
                </div>
            </div>
            <script>
                const chatLog = document.getElementById('chat-log');
                const questionInput = document.getElementById('question-input');
                const askForm = document.getElementById('ask-form');
                const statusNode = document.getElementById('status');
                const messagesList = document.getElementById('messages-list');
                const sendStopButton = document.getElementById('send-stop-button');
                const sendStopButtonLabel = sendStopButton.querySelector('.sr-only');
                const reasoningInput = document.getElementById('reasoning-effort');
                const reasoningToggle = document.getElementById('reasoning-toggle');
                const reasoningLabelNode = document.getElementById('reasoning-label');
                const reasoningDropdown = document.querySelector('.reasoning-dropdown');
                const reasoningMenu = document.getElementById('reasoning-menu');
                const reasoningOptions = reasoningMenu.querySelectorAll('.reasoning-option');
                const cancelEditButton = document.getElementById('cancel-edit');
                const showMoreButton = document.getElementById('show-more');
                const messagesStatus = document.getElementById('messages-status');
                const cachedMessages = {safe_cache_json};
                const totalCached = cachedMessages.length;
                const PAGE_SIZE = {MESSAGE_LIST_LIMIT};
                const MAX_LIMIT = totalCached;
                let currentLimit = totalCached ? Math.min(PAGE_SIZE, totalCached) : 0;
                let lastRenderedCount = 0;
                let conversation = [];
                let nextMessageId = 1;
                let editingMessageId = null;
                let activeRequest = null;
                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                const formatSeconds = (seconds) => Number(seconds).toFixed(1);

                function openReasoningMenu() {{
                    reasoningDropdown.dataset.open = 'true';
                    reasoningToggle.setAttribute('aria-expanded', 'true');
                }}

                function closeReasoningMenu() {{
                    reasoningDropdown.dataset.open = 'false';
                    reasoningToggle.setAttribute('aria-expanded', 'false');
                }}

                function setReasoning(value, label, button) {{
                    reasoningInput.value = value;
                    reasoningLabelNode.textContent = label;
                    reasoningOptions.forEach((option) => {{
                        option.classList.toggle('active', option === button);
                    }});
                    closeReasoningMenu();
                }}

                reasoningToggle.addEventListener('click', () => {{
                    const isOpen = reasoningDropdown.dataset.open === 'true';
                    if (isOpen) {{
                        closeReasoningMenu();
                    }} else {{
                        openReasoningMenu();
                    }}
                }});

                reasoningOptions.forEach((button) => {{
                    button.addEventListener('click', () => {{
                        setReasoning(button.dataset.value || '', button.dataset.label || button.textContent, button);
                    }});
                }});

                document.addEventListener('click', (event) => {{
                    if (!reasoningDropdown.contains(event.target)) {{
                        closeReasoningMenu();
                    }}
                }});

                document.addEventListener('keydown', (event) => {{
                    if (event.key === 'Escape') {{
                        closeReasoningMenu();
                    }}
                }});

                function renderConversation() {{
                    chatLog.innerHTML = '';
                    conversation.forEach((entry) => {{
                        const li = document.createElement('li');
                        li.dataset.id = entry.id;
                        li.classList.add(entry.role === 'user' ? 'question' : 'answer');
                        if (entry.pending) {{
                            li.classList.add('pending');
                        }}
                        if (entry.id === editingMessageId) {{
                            li.classList.add('editing');
                        }}

                        const header = document.createElement('div');
                        header.className = 'message-header';
                        const label = document.createElement('span');
                        label.textContent = entry.role === 'user' ? 'You' : 'Assistant';
                        header.appendChild(label);

                        if (entry.role === 'user') {{
                            const actionButton = document.createElement('button');
                            actionButton.type = 'button';
                            actionButton.className = 'message-action';
                            actionButton.textContent = entry.id === editingMessageId ? 'Editing…' : 'Edit';
                            actionButton.disabled = entry.id === editingMessageId;
                            actionButton.addEventListener('click', () => enterEditMode(entry.id));
                            header.appendChild(actionButton);
                        }}
                        if (entry.role === 'assistant' && entry.thinkTime) {{
                            const timing = document.createElement('span');
                            timing.className = 'message-timing';
                            timing.textContent = `Thought for ${{formatSeconds(entry.thinkTime)}} seconds`;
                            header.appendChild(timing);
                        }}

                        li.appendChild(header);
                        const body = document.createElement('p');
                        body.textContent = entry.content;
                        li.appendChild(body);
                        chatLog.appendChild(li);
                    }});
                    chatLog.scrollTop = chatLog.scrollHeight;
                }}

                function updateSendIntentLabel() {{
                    if (activeRequest) {{
                        return;
                    }}
                    const label = editingMessageId ? 'Resend edited question' : 'Send question';
                    sendStopButton.dataset.intent = editingMessageId ? 'resend' : 'send';
                    sendStopButton.setAttribute('aria-label', label);
                    sendStopButtonLabel.textContent = label;
                }}

                function enterEditMode(messageId) {{
                    const entry = conversation.find((msg) => msg.id === messageId && msg.role === 'user');
                    if (!entry) {{
                        return;
                    }}
                    if (activeRequest) {{
                        stopActiveRequest();
                    }}
                    editingMessageId = messageId;
                    questionInput.value = entry.content;
                    questionInput.focus();
                    cancelEditButton.hidden = false;
                    statusNode.textContent = 'Editing previous question. Sending will discard replies after it.';
                    updateSendIntentLabel();
                    renderConversation();
                }}

                function exitEditMode({{ silent = false }} = {{}}) {{
                    editingMessageId = null;
                    cancelEditButton.hidden = true;
                    if (!silent && !activeRequest) {{
                        statusNode.textContent = 'Ready.';
                    }}
                    updateSendIntentLabel();
                    renderConversation();
                }}

                cancelEditButton.addEventListener('click', () => {{
                    questionInput.value = '';
                    exitEditMode();
                }});

                function updateMessage(messageId, updates) {{
                    const index = conversation.findIndex((msg) => msg.id === messageId);
                    if (index === -1) {{
                        return;
                    }}
                    conversation[index] = {{ ...conversation[index], ...updates }};
                    renderConversation();
                }}

                function buildConversationFor(question) {{
                    if (editingMessageId !== null) {{
                        const editIndex = conversation.findIndex((msg) => msg.id === editingMessageId);
                        if (editIndex !== -1) {{
                            conversation[editIndex].content = question;
                            conversation = conversation.slice(0, editIndex + 1);
                        }}
                        exitEditMode({{ silent: true }});
                        return;
                    }}
                    conversation.push({{ id: nextMessageId++, role: 'user', content: question }});
                    renderConversation();
                }}

                function addAssistantPlaceholder() {{
                    const message = {{ id: nextMessageId++, role: 'assistant', content: 'Thinking...', pending: true }};
                    conversation.push(message);
                    renderConversation();
                    return message.id;
                }}

                function setRequestState(isActive) {{
                    if (isActive) {{
                        sendStopButton.dataset.mode = 'stop';
                        sendStopButton.type = 'button';
                        sendStopButton.setAttribute('aria-label', 'Stop response');
                        sendStopButtonLabel.textContent = 'Stop response';
                    }} else {{
                        sendStopButton.dataset.mode = 'send';
                        sendStopButton.type = 'submit';
                        updateSendIntentLabel();
                    }}
                }}

                function stopActiveRequest() {{
                    if (!activeRequest) {{
                        return;
                    }}
                    activeRequest.controller.abort();
                    statusNode.textContent = 'Stopping response...';
                }}

                sendStopButton.addEventListener('click', (event) => {{
                    if (sendStopButton.dataset.mode === 'stop') {{
                        event.preventDefault();
                        stopActiveRequest();
                    }}
                }});

                async function askQuestion(question, effort, signal) {{
                    const params = new URLSearchParams({{ question }});
                    if (effort) {{
                        params.append('reasoning', effort);
                    }}
                    const response = await fetch(`/ask?${{params.toString()}}`, {{ signal }});
                    if (!response.ok) {{
                        const data = await response.json().catch(() => ({{ detail: response.statusText }}));
                        throw new Error(data.detail || 'Failed to retrieve answer.');
                    }}
                    return response.json();
                }}

                async function loadMessages({{ showSpinner = false }} = {{}}) {{
                    if (!totalCached) {{
                        messagesList.innerHTML = '<p style="color:#64748b;">No cached messages are available yet.</p>';
                        showMoreButton.disabled = true;
                        messagesStatus.textContent = 'Messages will appear once the cache is populated.';
                        return;
                    }}

                    if (showSpinner) {{
                        messagesStatus.textContent = 'Loading cached messages...';
                    }} else if (!lastRenderedCount) {{
                        messagesStatus.textContent = 'Rendering cached messages...';
                    }}

                    const sliceEnd = Math.min(currentLimit, totalCached);
                    const items = cachedMessages.slice(0, sliceEnd);
                    messagesList.innerHTML = items.map(item => `
                        <div class="message-card">
                            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                                <strong>${{item.user_name}}</strong>
                                <time style="font-size:0.85rem;color:#94a3b8;">${{new Date(item.timestamp).toLocaleString()}}</time>
                            </div>
                            <p>${{item.message}}</p>
                        </div>
                    `).join('');
                    updateShowMoreState(items.length);
                    messagesStatus.textContent = '';
                }}

                function updateShowMoreState(renderedCount) {{
                    const previouslyRendered = lastRenderedCount;
                    lastRenderedCount = renderedCount;
                    const cannotGrow =
                        MAX_LIMIT === 0 ||
                        currentLimit >= MAX_LIMIT ||
                        renderedCount >= totalCached;
                    if (cannotGrow) {{
                        showMoreButton.disabled = true;
                        showMoreButton.textContent = 'No more messages';
                        messagesStatus.textContent = 'All available messages are displayed.';
                    }} else if (previouslyRendered === renderedCount && renderedCount !== 0) {{
                        showMoreButton.disabled = false;
                        showMoreButton.textContent = `Show ${{PAGE_SIZE}} more`;
                        showMoreButton.classList.add('pulse');
                        setTimeout(() => showMoreButton.classList.remove('pulse'), 300);
                        messagesStatus.textContent = 'No additional messages were found.';
                    }} else {{
                        showMoreButton.disabled = false;
                        showMoreButton.textContent = `Show ${{PAGE_SIZE}} more`;
                        messagesStatus.textContent = '';
                    }}
                }}

                showMoreButton.addEventListener('click', async () => {{
                    if (currentLimit >= MAX_LIMIT) return;
                    showMoreButton.disabled = true;
                    showMoreButton.classList.add('loading');
                    showMoreButton.textContent = 'Loading cached messages...';
                    await sleep(220);
                    currentLimit = Math.min(currentLimit + PAGE_SIZE, MAX_LIMIT);
                    await loadMessages({{ showSpinner: true }});
                    showMoreButton.classList.remove('loading');
                }});

                askForm.addEventListener('submit', async (event) => {{
                    event.preventDefault();
                    const question = questionInput.value.trim();
                    const reasoningEffort = reasoningInput.value;
                    if (!question) {{
                        statusNode.textContent = 'Enter a question first.';
                        return;
                    }}
                    if (activeRequest) {{
                        statusNode.textContent = 'Stop the current response before sending another prompt.';
                        return;
                    }}
                    buildConversationFor(question);
                    questionInput.value = '';
                    questionInput.focus();

                    const assistantId = addAssistantPlaceholder();
                    const controller = new AbortController();
                    const startedAt = performance.now();
                    activeRequest = {{ controller, assistantId, startedAt }};
                    setRequestState(true);
                    statusNode.textContent = 'Thinking...';

                    try {{
                        const result = await askQuestion(question, reasoningEffort, controller.signal);
                        const elapsed = Math.max(0.1, (performance.now() - startedAt) / 1000);
                        updateMessage(assistantId, {{
                            content: result.answer,
                            pending: false,
                            thinkTime: elapsed
                        }});
                        statusNode.textContent = `Used ${{result.sources_used}} messages.`;
                    }} catch (error) {{
                        if (error.name === 'AbortError') {{
                            updateMessage(assistantId, {{ content: 'Response stopped.', pending: false }});
                            statusNode.textContent = 'Response stopped.';
                        }} else {{
                            updateMessage(assistantId, {{ content: `Error: ${{error.message}}`, pending: false }});
                            statusNode.textContent = 'Unable to fetch answer.';
                        }}
                    }} finally {{
                        if (activeRequest && activeRequest.assistantId === assistantId) {{
                            activeRequest = null;
                            setRequestState(false);
                            if (!editingMessageId && !['Response stopped.', 'Unable to fetch answer.'].includes(statusNode.textContent) && !statusNode.textContent.startsWith('Used ')) {{
                                statusNode.textContent = 'Ready.';
                            }}
                        }}
                    }}
                }});

                questionInput.addEventListener('keydown', (event) => {{
                    if (event.key === 'Escape' && !cancelEditButton.hidden) {{
                        cancelEditButton.click();
                    }}
                }});

                updateSendIntentLabel();
                loadMessages();
            </script>

        </body>
    </html>
    """
    return HTMLResponse(content=html)


__all__ = ["app"]
