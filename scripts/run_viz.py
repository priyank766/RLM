"""
RLM Visualization Server — run RLM and watch every step in your browser.

Usage:
    uv run python scripts/run_viz.py
    uv run python scripts/run_viz.py --port 8765
    uv run python scripts/run_viz.py --model qwen3.5:2b --base-url http://localhost:11434

Then open http://localhost:8765 in your browser.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm.clients.ollama import OllamaClient
from rlm.viz_engine import run_rlm_observable


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="RLM Visualizer")

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

# Store references so WebSocket handler can access
_ollama_model = "qwen3.5:2b"
_ollama_url = "http://localhost:11434"


@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    html = FRONTEND_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection for real-time RLM events."""
    await websocket.accept()

    async def emit(event: dict):
        """Send an event to the frontend."""
        await websocket.send_json(event)

    try:
        while True:
            # Wait for a run request from the frontend
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("action") != "run":
                await emit({"type": "error", "message": f"Unknown action: {msg.get('action')}"})
                continue

            context = msg.get("context", "")
            query = msg.get("query", "")

            if not context or not query:
                await emit({"type": "error", "message": "Context and query are required."})
                continue

            # Create client
            client = OllamaClient(model=_ollama_model, base_url=_ollama_url)

            # Check if Ollama is available
            if not client.is_available():
                await emit({
                    "type": "error",
                    "message": f"Ollama is not running at {_ollama_url}. Start it with: ollama serve",
                })
                continue

            # Run the observable RLM in a background thread (LLM calls are blocking)
            try:
                await asyncio.to_thread(
                    _run_sync_wrapper,
                    client, context, query, emit
                )
            except Exception as e:
                await emit({"type": "error", "message": f"RLM run failed: {e}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


def _run_sync_wrapper(client, context, query, emit):
    """
    Bridge between sync LLM calls and async WebSocket emit.

    We run the async run_rlm_observable in a new event loop within the thread.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            run_rlm_observable(
                root_client=client,
                sub_client=client,
                context=context,
                query=query,
                emit=emit,
                max_iterations=20,
            )
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RLM Visualization Server")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--model", default="qwen3.5:2b", help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434",
                        help="Ollama server URL")
    args = parser.parse_args()

    global _ollama_model, _ollama_url
    _ollama_model = args.model
    _ollama_url = args.base_url

    # Check frontend exists
    if not FRONTEND_PATH.exists():
        print(f"ERROR: Frontend not found at {FRONTEND_PATH}")
        sys.exit(1)

    print(f"")
    print(f"  ╔══════════════════════════════════════════╗")
    print(f"  ║       RLM Visualizer  ⚡                 ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  URL:   http://{args.host}:{args.port}         ║")
    print(f"  ║  Model: {args.model:<33}║")
    print(f"  ║  Ollama: {args.base_url:<32}║")
    print(f"  ╚══════════════════════════════════════════╝")
    print(f"")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
