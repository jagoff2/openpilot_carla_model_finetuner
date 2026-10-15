"""Reversible external review holds; accepted artifacts and checkpoints stay immutable."""
from __future__ import annotations

import json
from pathlib import Path


class ReviewHoldError(RuntimeError):
    """Data or checkpoint lineage requires review before it can be used."""


def active_holds(corpus):
    """Read fresh on every check so in-memory archives cannot bypass a new hold."""
    path = Path(corpus) / "reports" / "review-holds.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise ReviewHoldError(f"cannot verify dataset review holds: {path}: {error}") from error
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(document.get("episodes"), dict)):
        raise ReviewHoldError(f"invalid dataset review-hold registry: {path}")
    held = {}
    seen = set()
    for episode, entry in document["episodes"].items():
        if (not isinstance(episode, str) or not episode or episode in (".", "..")
                or "/" in episode or "\\" in episode or not isinstance(entry, dict)
                or entry.get("status") not in ("held", "released")):
            raise ReviewHoldError(f"invalid dataset review-hold entry: {path}: {episode}")
        identity = episode.casefold()
        if identity in seen:
            raise ReviewHoldError(f"duplicate dataset review-hold identity: {path}: {episode}")
        seen.add(identity)
        if entry["status"] == "held":
            reasons = entry.get("reasons")
            if (not isinstance(reasons, list) or not reasons
                    or not all(isinstance(reason, str) and reason.strip() for reason in reasons)):
                raise ReviewHoldError(f"review hold requires reasons: {path}: {episode}")
            held[identity] = entry
    return held


def episode_name(manifest):
    """Normalize stored Windows or POSIX manifest paths independent of host OS."""
    parts = str(manifest).replace("\\", "/").rstrip("/").split("/")
    return parts[-2] if parts[-1] == "manifest.json" and len(parts) > 1 else parts[-1]


def assert_episode_not_held(manifest, *, corpus=None):
    path = Path(manifest).resolve()
    if corpus is None:
        # Development/test fixtures outside a corpus have no external registry.
        parent = path.parent if path.name == "manifest.json" else path
        if parent.parent.name.casefold() != "accepted":
            return
        corpus = parent.parent.parent
    name = episode_name(manifest)
    entry = active_holds(corpus).get(name.casefold())
    if entry:
        raise ReviewHoldError(f"dataset review hold for {name}: {'; '.join(entry['reasons'])}")


def checkpoint_review_findings(corpus, state):
    """Compare committed training counts to held episodes, without loading weights."""
    holds = active_holds(corpus)
    if not holds:
        return []
    known = state.get("known", {})
    by_hash = {row.get("sha256"): row for row in known.values() if isinstance(row, dict)}
    findings = []
    for digest, count in state.get("counts", {}).items():
        if count <= 0:
            continue
        row = by_hash.get(digest)
        if row is None:
            raise ReviewHoldError("checkpoint training lineage cannot be checked against active review holds")
        name = episode_name(row.get("id", row.get("path", "")))
        if not name:
            raise ReviewHoldError("checkpoint training lineage lacks an episode identity")
        if name.casefold() in holds:
            findings.append(dict(episode=name, manifest_sha256=digest, passes=count,
                                 reasons=list(holds[name.casefold()]["reasons"])))
    return findings


def assert_checkpoint_not_held(corpus, state):
    findings = checkpoint_review_findings(corpus, state)
    if findings:
        names = ", ".join(row["episode"] for row in findings)
        raise ReviewHoldError(f"checkpoint review hold: trained on held episodes: {names}; checkpoint preserved")
