from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConditionBatch:
    """Container passed to denoisers and pipelines for conditional generation."""

    values: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


class NullConditioner:
    """Default no-condition provider for unconditional generation."""

    def __call__(self, batch: dict[str, Any] | None = None) -> dict[str, Any] | None:
        del batch
        return None

