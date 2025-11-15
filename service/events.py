from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ProgressHub:
    """
    Tracks WebSocket subscribers per task and broadcasts progress updates.
    """

    def __init__(self):
        self._task_to_ws: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._task_to_ws.setdefault(task_id, set()).add(websocket)

    async def disconnect(self, task_id: str, websocket: WebSocket):
        async with self._lock:
            try:
                if task_id in self._task_to_ws:
                    self._task_to_ws[task_id].discard(websocket)
                    if not self._task_to_ws[task_id]:
                        del self._task_to_ws[task_id]
            except Exception:
                pass

    async def broadcast(self, task_id: str, message: dict):
        data = json.dumps(message)
        async with self._lock:
            sockets = list(self._task_to_ws.get(task_id, set()))
        for ws in sockets:
            try:
                await ws.send_text(data)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")


hub = ProgressHub()


