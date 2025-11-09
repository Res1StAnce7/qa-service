"""FastAPI entrypoint for the QA service."""
from __future__ import annotations

import json

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
)

app = FastAPI(title=settings.app_name)
MESSAGE_LIST_LIMIT = 50
MAX_MESSAGE_LIMIT = min(1000, settings.messages_api.limit)


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
                .cta-row {{
                    margin-top: 32px;
                    display: flex;
                    justify-content: center;
                    gap: 16px;
                }}
                .cta {{
                    padding: 14px 26px;
                    border-radius: 999px;
                    border: none;
                    background: #fff;
                    color: #1d4ed8;
                    font-weight: 600;
                    text-decoration: none;
                    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.15);
                }}
                .cta.secondary {{
                    background: rgba(255,255,255,0.15);
                    color: #fff;
                    border: 1px solid rgba(255,255,255,0.3);
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
                <div class="cta-row">
                    <a class="cta" href="/ask?question=What%20are%20Amira's%20favorite%20restaurants?">Quick Ask</a>
                    <a class="cta secondary" href="/demo">Interactive Demo</a>
                </div>
            </header>
            <main>
                <section class="card-grid">
                    <div class="card">
                        <h3>Ask Endpoint</h3>
                        <p>Call <code>/ask?question=...</code> for instantaneous answers grounded in the latest concierge messages.</p>
                        <a href="/ask?question=Who%20needs%20a%20payment%20check">Try sample query →</a>
                    </div>
                    <div class="card">
                        <h3>Message Explorer</h3>
                        <p>Use <code>/messages?limit=50</code> to fetch the raw feed and plug it into your own tooling.</p>
                        <a href="/messages">View messages →</a>
                    </div>
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
) -> AnswerResponse:
    normalized = _normalize_question(question)
    return await _answer_question(normalized)


def _normalize_question(question: str) -> str:
    value = question.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    if not value:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    return value


async def _answer_question(question: str) -> AnswerResponse:
    try:
        answer, count = await qa_service.answer_question(question)
        return AnswerResponse(answer=answer, sources_used=count)
    except Exception as exc:  # pragma: no cover - FastAPI will handle logging
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/demo", response_class=HTMLResponse)
async def demo() -> HTMLResponse:
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
                    gap: 10px;
                }}
                .input-row input {{
                    flex: 1;
                    padding: 14px 16px;
                    border-radius: 12px;
                    border: 1px solid var(--border);
                    font-size: 1rem;
                    transition: border 0.2s, box-shadow 0.2s;
                }}
                .input-row input:focus {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
                }}
                .input-row button {{
                    padding: 0 20px;
                    border: none;
                    border-radius: 12px;
                    background: var(--primary);
                    color: white;
                    font-weight: 600;
                    cursor: pointer;
                    transition: background 0.2s, transform 0.1s;
                }}
                .input-row button:hover {{
                    background: var(--primary-dark);
                }}
                .input-row button:active {{
                    transform: translateY(1px);
                }}
                .input-row button:disabled {{
                    opacity: 0.6;
                    cursor: not-allowed;
                    transform: none;
                }}
                .status {{
                    min-height: 22px;
                    color: #475569;
                    margin-top: 10px;
                    font-size: 0.95rem;
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
            </style>
        </head>
        <body>
            <div class="container">
                <div class="panel chat-panel">
                    <h2>Ask Concierge</h2>
                    <ul id="chat-log" class="chat-log"></ul>
                    <form id="ask-form" class="input-row">
                        <input id="question-input" type="text" placeholder="Ask anything about the members..."
                            autocomplete="off" />
                        <button type="submit">Ask</button>
                    </form>
                    <div id="status" class="status">Ready.</div>
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
                const askButton = askForm.querySelector('button');
                const showMoreButton = document.getElementById('show-more');
                const messagesStatus = document.getElementById('messages-status');
                const PAGE_SIZE = {MESSAGE_LIST_LIMIT};
                const MAX_LIMIT = {MAX_MESSAGE_LIMIT};
                let currentLimit = PAGE_SIZE;
                let lastRenderedCount = 0;

                function addChatBubble(text, role) {{
                    const li = document.createElement('li');
                    li.classList.add(role);
                    li.textContent = text;
                    chatLog.appendChild(li);
                    chatLog.scrollTop = chatLog.scrollHeight;
                }}

                async function askQuestion(question) {{
                    const url = `/ask?question=${{encodeURIComponent(question)}}`;
                    const response = await fetch(url);
                    if (!response.ok) {{
                        const data = await response.json().catch(() => ({{ detail: response.statusText }}));
                        throw new Error(data.detail || 'Failed to retrieve answer.');
                    }}
                    return response.json();
                }}

                async function loadMessages({{ showSpinner = false }} = {{}}) {{
                    if (showSpinner) {{
                        messagesStatus.textContent = 'Loading more messages...';
                    }} else if (!lastRenderedCount) {{
                        messagesStatus.textContent = 'Loading messages...';
                    }}
                    try {{
                        const response = await fetch(`/messages?limit=${{currentLimit}}`);
                        if (!response.ok) {{
                            throw new Error('Failed to load messages');
                        }}
                        const items = await response.json();
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
                    }} catch (error) {{
                        messagesList.innerHTML = `<p style=\"color:#b91c1c;\">Unable to load messages: ${{error.message}}</p>`;
                        messagesStatus.textContent = 'Unable to load messages right now.';
                        showMoreButton.disabled = true;
                    }}
                }}

                function updateShowMoreState(renderedCount) {{
                    const previouslyRendered = lastRenderedCount;
                    lastRenderedCount = renderedCount;
                    const cannotGrow = currentLimit >= MAX_LIMIT || renderedCount < currentLimit;
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
                    showMoreButton.textContent = 'Loading...';
                    currentLimit = Math.min(currentLimit + PAGE_SIZE, MAX_LIMIT);
                    await loadMessages({{ showSpinner: true }});
                    showMoreButton.classList.remove('loading');
                }});

                askForm.addEventListener('submit', async (event) => {{
                    event.preventDefault();
                    const question = questionInput.value.trim();
                    if (!question) {{
                        statusNode.textContent = 'Enter a question first.';
                        return;
                    }}
                    questionInput.value = '';
                    questionInput.focus();
                    addChatBubble(question, 'question');
                    statusNode.textContent = 'Thinking...';
                    askButton.disabled = true;
                    try {{
                        const result = await askQuestion(question);
                        addChatBubble(result.answer, 'answer');
                        statusNode.textContent = `Used ${{result.sources_used}} messages.`;
                    }} catch (error) {{
                        addChatBubble(`Error: ${{error.message}}`, 'answer');
                        statusNode.textContent = 'Unable to fetch answer.';
                    }} finally {{
                        askButton.disabled = false;
                    }}
                }});

                loadMessages();
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


__all__ = ["app"]
