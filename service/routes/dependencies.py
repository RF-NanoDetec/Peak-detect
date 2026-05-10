from __future__ import annotations

from fastapi import HTTPException, Request

from service.jobs import TaskManager
from service.storage import InMemoryStore


def get_store(request: Request) -> InMemoryStore:
    store = getattr(request.app.state, "store", None)
    if not store:
        raise HTTPException(status_code=500, detail="Storage not initialized")
    return store


def get_task_manager(request: Request) -> TaskManager:
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        raise HTTPException(status_code=500, detail="Task manager not initialized")
    return task_manager
