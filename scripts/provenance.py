"""
scripts/provenance.py
=====================
A stamp for every report under reports/: the command that wrote it, the commit it
ran at and the engine fingerprint, so a reader can tell which version of the model
a report measured and how to produce it again. The fingerprint is the one
scripts/check_cache_consistency.py records in the published manifest, so a report
whose engine matches the manifest's measured the model the site shows.
"""

import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stamp() -> dict:
    """The command, commit and engine behind the report being written now."""
    from scripts.check_cache_consistency import engine_fingerprint

    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(ROOT)
    except ValueError:
        pass
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "command": shlex.join(["python", str(script), *sys.argv[1:]]),
        "commit": _git("rev-parse", "--short=12", "HEAD"),
        "uncommitted_changes": bool(status),
        "engine": engine_fingerprint(),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def markdown(s: dict) -> str:
    commit = f"commit `{s['commit']}`" if s["commit"] else "an unknown commit"
    if s["uncommitted_changes"]:
        commit += " with uncommitted changes"
    return (f"_Written by `{s['command']}` on {s['generated']}, at {commit}; "
            f"engine `{s['engine']}`._")


def stamp_lines(lines: list, s: dict) -> list:
    """Put the stamp under a report's title, which is lines[0]."""
    lines[1:1] = ["", markdown(s)]
    return lines
