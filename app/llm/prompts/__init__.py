"""Prompt version registry.

Each version is a separate, immutable module (v1.py, v2.py, ...) exposing
VERSION, SYSTEM_PROMPT, and build_user_prompt(). This module is the single
place that knows how a version string maps to one -- callers (the summarize
service, tests) ask for a version by name and never import vN modules
directly, so adding v2 never requires touching call sites.
"""

from types import ModuleType

from app.llm.prompts import v1

_VERSIONS: dict[str, ModuleType] = {
    v1.VERSION: v1,
}

DEFAULT_VERSION = v1.VERSION


def get_prompt_module(version: str = DEFAULT_VERSION) -> ModuleType:
    try:
        return _VERSIONS[version]
    except KeyError:
        raise ValueError(f"Unknown prompt version {version!r}. Available: {sorted(_VERSIONS)}") from None
