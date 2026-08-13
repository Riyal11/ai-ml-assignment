"""Abstract MCP-style connector to an enterprise document store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DocumentConnector(ABC):
    """Store-agnostic document connector (SharePoint / S3 / ERP)."""

    @abstractmethod
    def fetch(self, doc_id: str) -> bytes | str:
        """Fetch document content by identifier."""

    @abstractmethod
    def write(self, doc_id: str, invoice_json: dict[str, Any], content_hash: str) -> bool:
        """Persist extraction result idempotently."""

    @abstractmethod
    def list_pending(self, limit: int = 100) -> list[dict[str, Any]]:
        """List documents awaiting extraction."""
