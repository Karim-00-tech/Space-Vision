# server.py
import os
import json
import uuid
import asyncio
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv()

# Gemini client
from google import genai

API_KEY = "AIzaSyA6lFrRgEsqjd7OKsXfEWJXOBlQRPlnRaY"
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment")

# Initialize genai client with explicit key
client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-flash"   # adjust if needed

app = FastAPI()

# Allow your frontend origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:5500", "http://localhost:3000", "*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (for demo). Use Redis or DB in production.
class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context: Dict[str, Any] = {}          # latest UI context snapshot
        self.history: List[Dict[str,str]] = []     # chat history [{'role':'user','text':...}, {'role':'assistant','text':...}]
        self.last_active = asyncio.get_event_loop().time()

sessions: Dict[str, Session] = {}

# Simple helper to create or get a session
def get_or_create_session(session_id: str | None) -> Session:
    if session_id and session_id in sessions:
        s = sessions[session_id]
    else:
        new_id = session_id or str(uuid.uuid4())
        s = Session(new_id)
        sessions[new_id] = s
    s.last_active = asyncio.get_event_loop().time()
    return s

# Build a "system" prompt using UI context so the model knows the scene
def build_prompt_from_context(context: Dict[str,Any], user_message: str) -> str:
    parts = []
    parts.append("You are an assistant embedded in a NASA dataset explorer web app.")
    parts.append("Be concise, helpful, and refer to the current viewer context when relevant.")
    if context:
        # Add dataset info if present
        dataset = context.get("dataset")
        if dataset:
            parts.append(f"Dataset: {dataset.get('title','N/A')} (id: {dataset.get('id','')})")
            if dataset.get("description"):
                parts.append(f"Description: {dataset.get('description')}")
            if dataset.get("source"):
                parts.append(f"Source: {dataset.get('source')}")
        # viewer state
        viewer = context.get("viewer")
        if viewer:
            parts.append(f"Viewer zoom: {viewer.get('zoom')}, position: {viewer.get('position')}")
        # coordinates and markers
        coords = context.get("coordinates")
        if coords and coords.get("x") is not None:
            parts.append(f"Last pointer coordinates (norm): x={coords.get('x')}, y={coords.get('y')}")
        markers = context.get("markers")
        if markers:
            # keep markers short
            short_markers = []
            for m in markers[:8]:
                short_markers.append(f"[{m.get('id')}] {m.get('text', '')} @({m.get('x'):.3f},{m.get('y'):.3f})")
            parts.append("Markers: " + "; ".join(short_markers))
    # Now the user message
    parts.append("User message: " + user_message)
    # Instructions for assistant reply
    parts.append("When relevant, reference dataset and coordinates. Keep replies short (1-3 sentences) unless asked otherwise.")
    prompt = "\n\n".join(parts)
    return prompt

# WebSocket endpoint for assistant
@app.websocket("/ws/assistant")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session: Session | None = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type":"error", "text":"Invalid JSON payload"}))
                continue

            typ = payload.get("type")
            # INIT: client provides stored session_id or null
            if typ == "init":
                client_session_id = payload.get("session_id")
                session = get_or_create_session(client_session_id)
                # respond with assigned session_id so client can save it
                await ws.send_text(json.dumps({"type":"init", "session_id": session.session_id}))
                continue

            # Ensure session exists
            if session is None:
                session = get_or_create_session(None)
                await ws.send_text(json.dumps({"type":"init", "session_id": session.session_id}))

            # CONTEXT: UI sends context snapshots
            if typ == "context":
                ctx = payload.get("context") or {}
                session.context = ctx
                session.last_active = asyncio.get_event_loop().time()
                # optionally ack
                # await ws.send_text(json.dumps({"type":"ack", "text":"context received"}))
                continue

            # MESSAGE: main user question
            if typ == "message":
                user_text = payload.get("text","").strip()
                if not user_text:
                    await ws.send_text(json.dumps({"type":"error", "text":"Empty message"}))
                    continue

                # Append to session history
                session.history.append({"role":"user","text":user_text})
                session.last_active = asyncio.get_event_loop().time()

                # Build prompt with context
                prompt = build_prompt_from_context(session.context, user_text)

                # Call Gemini / GenAI
                try:
                    # Optionally, implement simple throttling / concurrency control here
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=prompt,
                    )
                    # gemini's response text may be in resp.text
                    text = resp.text if hasattr(resp, "text") else str(resp)
                    # save assistant reply to history
                    session.history.append({"role":"assistant","text":text})
                    # Send it back to the front-end
                    await ws.send_text(json.dumps({"type":"response", "text": text}))
                except Exception as e:
                    # On error, log and inform client
                    err_msg = f"Model error: {type(e).__name__} - {str(e)}"
                    await ws.send_text(json.dumps({"type":"error", "text": err_msg}))
                continue

            # Unknown type
            await ws.send_text(json.dumps({"type":"error", "text": f"Unknown event type: {typ}"}))

    except WebSocketDisconnect:
        # client disconnected
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type":"error", "text": f"Server error: {str(e)}"}))
        except Exception:
            pass
        return