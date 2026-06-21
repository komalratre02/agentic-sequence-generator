"""
Prompt loader with versioning support.

Prompts live in app/prompts/<name>_v<N>.txt
Every load is logged with its version string.
"""
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str = "v1") -> str:
    """
    Load a prompt from disk.

    Args:
        name:    Prompt name, e.g. 'planner', 'writer', 'reviewer', 'research'
        version: Version string, e.g. 'v1', 'v2'

    Returns:
        Prompt text as a string.
    """
    filename = f"{name}_{version}.txt"
    path = PROMPTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    text = path.read_text(encoding="utf-8")
    logger.debug("Loaded prompt '%s' from %s", filename, path)
    return text


def prompt_version_tag(name: str, version: str = "v1") -> str:
    """Return a canonical version tag, e.g. 'writer_v1'."""
    return f"{name}_{version}"
