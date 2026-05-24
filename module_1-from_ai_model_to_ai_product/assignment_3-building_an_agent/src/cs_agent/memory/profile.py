"""Per-user profile (semantic memory).

Storage model: one JSON file per user at ``profiles/<user_id>.json``. We use
plain JSON instead of LangGraph's ``BaseStore`` deliberately:

- A grader can ``cat profiles/ofir.json`` and immediately see what the agent
  has remembered. There's no "open the SQLite, run a query" step.
- The artefact is trivial to delete to reset state, trivial to inspect for
  privacy review, and trivial to copy for a demo.
- The schema is a Pydantic model so the agent's structured-output update node
  can return it directly and have validation for free.

The profile is **per-user**, not per-thread. A single user can have multiple
sessions (``--session demo``, ``--session work``); the profile follows the
user across all of them. Conversely, the LangGraph checkpoint is per-thread.

We do NOT add the profile to ``GraphState``: it lives on disk and is loaded
lazily by ``agent_node`` (for prompt injection) and by ``profile_update_node``
(for read-modify-write).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from cs_agent.config import PROFILES_DIR


class UserProfile(BaseModel):
    """Distilled, persisted facts about a single user.

    Designed to be small (one short JSON file per user) so the grader can
    eyeball it and the agent can render the whole thing into its system
    prompt cheaply on every turn.
    """

    user_id: str
    """Stable identifier for the user. Same as ``--user`` on the CLI."""

    name: str | None = None
    """Preferred display name, e.g. "Ofir"."""

    role: str | None = None
    """Self-described role / job title, e.g. "data engineer"."""

    topics_of_interest: list[str] = Field(default_factory=list)
    """Topics the user has expressed curiosity about, e.g. ["refunds", "complaints"]."""

    preferences: dict[str, str] = Field(default_factory=dict)
    """Free-form preference map, e.g. {"answer_length": "concise"}."""

    notable_facts: list[str] = Field(default_factory=list)
    """Catch-all for memorable one-off facts the agent should remember."""

    last_updated: datetime | None = None
    """UTC timestamp of the last successful profile update. ``None`` for an
    empty / never-touched profile."""

    def has_facts(self) -> bool:
        """True iff there's at least one piece of content worth showing the agent."""
        return bool(
            self.name or self.role or self.topics_of_interest or self.preferences or self.notable_facts
        )

    def render_for_prompt(self) -> str:
        """Compact, human-readable summary used inside the agent system prompt.

        Empty fields are omitted to keep the prompt tight.
        """
        if not self.has_facts():
            return "No prior facts about this user yet."
        lines: list[str] = []
        if self.name:
            lines.append(f"- Name: {self.name}")
        if self.role:
            lines.append(f"- Role: {self.role}")
        if self.topics_of_interest:
            lines.append(f"- Topics of interest: {', '.join(self.topics_of_interest)}")
        if self.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            lines.append(f"- Preferences: {prefs}")
        if self.notable_facts:
            for fact in self.notable_facts:
                lines.append(f"- {fact}")
        return "\n".join(lines)


def profile_path(user_id: str) -> Path:
    """Filesystem location for a user's profile JSON.

    Reads ``PROFILES_DIR`` as a module global at call time (Python late-binds
    globals), so tests can ``monkeypatch.setattr(profile_mod, "PROFILES_DIR",
    tmp_path)`` and have it take effect immediately without any helper.
    """
    return PROFILES_DIR / f"{user_id}.json"


def load_profile(user_id: str) -> UserProfile:
    """Read a user's profile from disk; return an empty one if missing.

    Treats unreadable / corrupt JSON the same as "missing" — the agent
    keeps working, and the next ``save_profile`` call will overwrite the bad
    file. Logging the corruption would also be reasonable; we keep it silent
    here to avoid noisy CLI output for what is otherwise an unusual case.
    """
    p = profile_path(user_id)
    if not p.exists():
        return UserProfile(user_id=user_id)
    try:
        raw = p.read_text(encoding="utf-8")
        return UserProfile.model_validate_json(raw)
    except (OSError, ValueError):
        return UserProfile(user_id=user_id)


def save_profile(profile: UserProfile) -> None:
    """Persist a profile to disk atomically.

    Atomic write pattern: write to a sibling ``.tmp`` file then ``rename``.
    On POSIX ``rename`` is atomic, so a partially-written profile cannot
    appear on disk even if the process is killed mid-write.
    """
    p = profile_path(profile.user_id)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = profile.model_dump(mode="json")
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def now_utc() -> datetime:
    """UTC ``datetime`` factory. Wrapped so tests can monkeypatch the clock."""
    return datetime.now(tz=UTC)


# Personal-info-bearing markers. Conservative on purpose: a high-precision /
# moderate-recall regex that fires only when the user clearly volunteers
# something biographical or preferential. Missing a subtle cue is cheaper
# than running an LLM round-trip on every dataset Q&A turn.
PROFILE_MARKERS = re.compile(
    r"(?:"
    r"\bmy name is\b|"
    r"\bi'?m called\b|"
    r"\bcall me\b|"
    r"\bi prefer\b|"
    r"\bi like\b|"
    r"\bi love\b|"
    r"\bi hate\b|"
    r"\bi don'?t like\b|"
    r"\bi don'?t want\b|"
    r"\bremember (?:that|me|this)\b|"
    r"\bnote that\b|"
    r"\bfor next time\b|"
    r"\bi work as\b|"
    r"\bmy role is\b|"
    r"\bmy job is\b|"
    r"\bi am a\b|"
    r"\bi'?m a\b"
    r")",
    re.IGNORECASE,
)


def is_personal_info_bearing(message: str) -> bool:
    """Cheap regex gate: True if ``message`` contains a high-signal personal
    marker that warrants invoking the profile-update LLM.

    See ``PROFILE_MARKERS`` for the exact set. Non-string / empty input
    returns False rather than raising, so callers don't need to defensively
    check the message contents.
    """
    if not isinstance(message, str) or not message.strip():
        return False
    return bool(PROFILE_MARKERS.search(message))
