from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskManager:
    """
    Simple in-memory task manager that runs blocking callables in a thread pool
    and emits progress events via a provided callback.
    """

    def __init__(self, max_workers: Optional[int] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        func: Callable[..., Any],
        *,
        on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Submit a task. Returns taskId immediately.
        """
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {"status": TaskStatus.PENDING, "result": None, "error": None}

        def progress_callback(update: Dict[str, Any]):
            if on_progress:
                try:
                    on_progress(task_id, update)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        def run():
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.RUNNING
            logger.info(f"Task {task_id}: Starting execution")
            try:
                result = func(progress_callback=progress_callback, **kwargs)
                with self._lock:
                    self._tasks[task_id]["status"] = TaskStatus.COMPLETED
                    self._tasks[task_id]["result"] = result
                logger.info(f"Task {task_id}: Completed successfully, sending final progress update")
                # Send final completion message via progress callback
                progress_callback({"progress": 100, "status": "completed", "result": result})
            except Exception as exc:
                logger.exception(f"Task {task_id}: Failed with exception")
                with self._lock:
                    self._tasks[task_id]["status"] = TaskStatus.FAILED
                    self._tasks[task_id]["error"] = str(exc)
                # Send final failure message via progress callback
                progress_callback({"progress": 0, "status": "failed", "error": str(exc)})

        self._executor.submit(run)
        return task_id

    def get(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._tasks.get(task_id) or {})

    def cancel(self, task_id: str) -> bool:
        # Cooperative cancellation not implemented; placeholder for future
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id]["status"] in (TaskStatus.PENDING, TaskStatus.RUNNING):
                self._tasks[task_id]["status"] = TaskStatus.CANCELLED
                return True
        return False


