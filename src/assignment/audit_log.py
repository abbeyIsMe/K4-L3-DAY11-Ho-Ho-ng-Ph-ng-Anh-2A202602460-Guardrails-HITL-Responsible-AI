"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}          # request_id -> thời điểm bắt đầu (monotonic)
        self._inputs: dict[str, dict] = {}         # request_id -> {user_id, input, timestamp_in}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + thời điểm bắt đầu. Trả về request_id để ghép với record_output."""
        request_id = request_id or uuid.uuid4().hex[:8]
        self._open[request_id] = time.monotonic()
        self._inputs[request_id] = {
            "user_id": user_id,
            "input": text,
            "timestamp_in": utc_now_iso(),
        }
        return request_id

    def _find_open_request(self, user_id: str) -> str | None:
        """Request đang mở gần nhất của user (khi caller không truyền request_id)."""
        for rid in reversed(list(self._inputs)):
            if self._inputs[rid]["user_id"] == user_id:
                return rid
        return None

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, lớp quyết định, latency; append vào self.logs."""
        request_id = request_id or self._find_open_request(user_id)
        started = self._open.pop(request_id, None) if request_id else None
        inp = self._inputs.pop(request_id, None) if request_id else None

        entry = {
            "request_id": request_id,
            "user_id": user_id,
            "input": inp["input"] if inp else None,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "timestamp_in": inp["timestamp_in"] if inp else None,
            "timestamp_out": utc_now_iso(),
            "latency_ms": (
                round((time.monotonic() - started) * 1000, 2)
                if started is not None
                else None
            ),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()