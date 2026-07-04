"""Data models for Jira Scout."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Ticket:
    id: str
    title: str
    description: str
    status: str
    resolution: str = ""
    created: str = ""
    labels: List[str] = field(default_factory=list)

    @property
    def searchable_text(self) -> str:
        """Combined text used to build the ticket's vector representation."""
        return f"{self.title}. {self.description}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "resolution": self.resolution,
            "created": self.created,
            "labels": self.labels,
        }
