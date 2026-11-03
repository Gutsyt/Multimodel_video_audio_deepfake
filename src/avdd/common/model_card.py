"""Model attribution record, carried into every analysis result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from avdd.common.types import Modality


@dataclass(frozen=True)
class ModelCard:
    """Provenance, measured metrics and limitations of a detector."""

    name: str
    version: str
    modality: Modality
    license: str
    source_url: str
    citation: str
    trained_on: str
    limitations: tuple[str, ...]
    is_test_fixture: bool = False
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value
        d["limitations"] = list(self.limitations)
        return d
