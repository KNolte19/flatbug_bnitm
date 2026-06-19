from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ClassifierService(ABC):
    @abstractmethod
    def classify_segments(self, segments: list[Any]) -> list[Any]:
        raise NotImplementedError


class NoOpClassifierService(ClassifierService):
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def classify_segments(self, segments: list[Any]) -> list[Any]:
        return []
