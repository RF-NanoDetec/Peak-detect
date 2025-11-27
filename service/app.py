from __future__ import annotations

import logging
from typing import Dict

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Support running as a script (no package) and as a module/package
try:
    from .events import hub  # type: ignore[import-not-found]
    from .handlers import router as api_router  # type: ignore[import-not-found]
    from .jobs import TaskManager  # type: ignore[import-not-found]
    from .models import Params  # type: ignore[import-not-found]
    from .storage import InMemoryStore  # type: ignore[import-not-found]
except Exception:
    # Fallback to absolute imports when relative imports are not available
    from service.events import hub  # type: ignore[no-redef]
    from service.handlers import router as api_router  # type: ignore[no-redef]
    from service.jobs import TaskManager  # type: ignore[no-redef]
    from service.models import Params  # type: ignore[no-redef]
    from service.storage import InMemoryStore  # type: ignore[no-redef]

import asyncio
import sys
from pathlib import Path

from config import APP_VERSION

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Peak Analysis Local Service",
        version=APP_VERSION,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS restricted to localhost origins
    origins = [
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _on_startup():
        # Store the running loop for cross-thread progress dispatch
        app.state.loop = asyncio.get_running_loop()
        app.state.task_manager = TaskManager()
        app.state.store = InMemoryStore()
        app.state.params_default = Params()

    def broadcast_progress(task_id: str, update: dict):
        # Safe to call from worker threads: schedule coroutine on main loop
        try:
            asyncio.run_coroutine_threadsafe(
                hub.broadcast(task_id, update), app.state.loop
            )
        except Exception:
            logger.debug("Failed to schedule progress broadcast", exc_info=True)

    # Expose helper for handlers
    app.broadcast_progress = broadcast_progress  # type: ignore[attr-defined]

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/version")
    async def version() -> Dict[str, str]:
        return {"version": APP_VERSION}

    @app.websocket("/ws/progress")
    async def ws_progress(
        websocket: WebSocket, taskId: str = Query(..., alias="taskId")
    ):
        try:
            await hub.connect(taskId, websocket)
            # Keep the socket open; we don't receive messages, only send
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            try:
                await hub.disconnect(taskId, websocket)
            except Exception:
                pass
        except Exception:
            try:
                await hub.disconnect(taskId, websocket)
            except Exception:
                pass

    return app


app = create_app()
app.include_router(api_router)

# Mount static files from Next.js build (if available)
# Check for ui-web/out directory (Next.js static export)
# When running as PyInstaller bundle, use sys._MEIPASS
if getattr(sys, "frozen", False):
    # Running as compiled executable
    base_path = Path(sys._MEIPASS)
else:
    # Running as script
    base_path = Path(__file__).parent.parent

static_dir = base_path / "ui-web" / "out"
next_static_dir = static_dir / "_next"
if static_dir.exists() and next_static_dir.exists():
    logger.info(f"Serving static files from {static_dir}")

    # Mount static assets
    app.mount("/_next", StaticFiles(directory=str(next_static_dir)), name="next_static")

    # Explicit root route handler
    @app.get("/")
    async def serve_root():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        # Try Next.js App Router structure: app/page.html
        app_page_path = static_dir / "app" / "page.html"
        if app_page_path.exists():
            return FileResponse(app_page_path)
        # If neither exists, return helpful error
        return {
            "error": "Static files not found",
            "message": "The UI has not been built. Please run 'npm run build' in the ui-web directory.",
            "static_dir": str(static_dir),
        }

    # Serve index.html for all other routes (SPA fallback)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return {"error": "Not found"}

        # Try to serve the requested file
        file_path = static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # For HTML files or routes, try to find .html version
        html_path = static_dir / f"{full_path}.html"
        if html_path.is_file():
            return FileResponse(html_path)

        # Try Next.js App Router structure: app/{path}/page.html
        app_route_path = static_dir / "app" / full_path / "page.html"
        if app_route_path.exists():
            return FileResponse(app_route_path)

        # Fallback to index.html for SPA routing
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        return {"error": "Not found", "path": full_path}
else:
    logger.info(
        "Static files not found. UI must be served separately (e.g., npm run dev)"
    )

    # Add a root route handler when static files aren't available
    @app.get("/")
    async def root():
        return {
            "message": "Peak Analysis Service API",
            "status": "running",
            "docs": "/api/docs",
            "health": "/api/health",
            "note": "Static UI files not found. UI must be served separately (e.g., npm run dev in ui-web directory)",
            "static_dir": str(static_dir),
        }


# Optional entrypoint for local runs: `python -m service.app` or running the EXE
if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    # Open browser after a short delay to let the server start
    def open_browser():
        import time

        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8765")

    # Start browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    logger.info("Starting Peak Analysis Tool server...")
    logger.info("Server will be available at http://127.0.0.1:8765")
    logger.info("Browser will open automatically...")

    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False, log_level="info")
