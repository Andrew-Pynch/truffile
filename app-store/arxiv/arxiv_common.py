from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

MAX_RESULTS = 50
DEFAULT_STORAGE_PATH = Path.home() / ".arxiv-mcp-server" / "papers"
DEFAULT_BG_STATE_PATH = Path("/root/.arxiv-truffle/arxiv_bg_state.json")


def get_storage_path() -> Path:
    raw = str(os.getenv("ARXIV_STORAGE_PATH", "")).strip()
    path = Path(raw) if raw else DEFAULT_STORAGE_PATH
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_bg_state_path() -> Path:
    raw = str(os.getenv("ARXIV_BG_STATE_PATH", "")).strip()
    path = Path(raw) if raw else DEFAULT_BG_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_research_interests(raw: str | None) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ",")
    out: list[str] = []
    seen: set[str] = set()
    for part in normalized.split(","):
        interest = part.strip()
        if not interest:
            continue
        key = interest.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(interest)
    return out


# ---------------------------------------------------------------------------
# User config (interests + open questions) — persistent state file
# ---------------------------------------------------------------------------

DEFAULT_USER_CONFIG_PATH = Path("/root/.arxiv-truffle/user_config.json")


def get_user_config_path() -> Path:
    raw = str(os.getenv("ARXIV_USER_CONFIG_PATH", "")).strip()
    path = Path(raw) if raw else DEFAULT_USER_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_user_config() -> dict[str, Any]:
    path = get_user_config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"interests": [], "open_questions": []}


def save_user_config(config: dict[str, Any]) -> None:
    path = get_user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def get_effective_interests(env_raw: str) -> list[str]:
    """Get interests from user config file, falling back to env var."""
    config = load_user_config()
    config_interests = config.get("interests", [])
    if config_interests:
        return config_interests
    return parse_research_interests(env_raw)


def get_open_questions() -> list[dict[str, Any]]:
    config = load_user_config()
    return config.get("open_questions", [])


def add_open_question(question: str) -> list[dict[str, Any]]:
    config = load_user_config()
    questions = config.get("open_questions", [])
    questions.append({
        "question": question,
        "status": "open",
        "added": date.today().isoformat(),
        "resolved_by": None,
        "notes": None,
    })
    config["open_questions"] = questions
    save_user_config(config)
    return questions


def resolve_open_question(
    index: int,
    *,
    paper_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any] | None:
    config = load_user_config()
    questions = config.get("open_questions", [])
    if index < 0 or index >= len(questions):
        return None
    questions[index]["status"] = "answered"
    questions[index]["resolved_by"] = paper_id
    questions[index]["notes"] = notes
    config["open_questions"] = questions
    save_user_config(config)
    return questions[index]

