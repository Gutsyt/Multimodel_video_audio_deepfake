"""Common return type for per-modality analysis branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from avdd.common.types import ModalityEvidence


@dataclass(frozen=True)
class BranchOutput:
    """Validated evidence plus JSON-serializable details (segments, metrics, model card)."""

    evidence: ModalityEvidence
    details: dict[str, Any] = field(default_factory=dict)
